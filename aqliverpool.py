#!/usr/bin/python3

# coding: utf-8

from optparse import OptionParser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
from matplotlib.text import Annotation, Text
import numpy as np
import urllib
from urllib import request
import re
import html
import sys, os
import pickle
from datetime import datetime, timedelta
import tweepy
import sqlite3
import math

from collections import deque

apatch = None
urlstr = "https://uk-air.defra.gov.uk/latest/currentlevels?view=site#L"
shorturlstr = "https://goo.gl/ZpELjS"

urlWHO = "http://apps.who.int/iris/bitstream/10665/69477/1/WHO_SDE_PHE_OEH_06.02_eng.pdf"

sitename = b'Liverpool'

mgm3 = '\u03BCgm\u207B\u00B3'
O3, NO2, SO2, PM25, PM100 = "O\u2083", "NO\u2082", "SO\u2082", "PM\u2082\u2085", "PM\u2081\u2080\u2080"
guides = {O3:100, NO2:200, SO2:20, PM25:25, PM100:50} # source: http://apps.who.int/iris/bitstream/10665/69477/1/WHO_SDE_PHE_OEH_06.02_eng.pdf  
meansWHO = {O3:'8h', NO2:'1h', SO2:'10m', PM25:'24h', PM100:'24h'}
meansDEFRA = {O3:'8h', NO2:'1h', SO2:'max 15m', PM25:'24h', PM100:'24h'}
consumer_key, consumer_secret, access_token, access_token_secret = None, None, None, None

def loadAPIKeys():
    global consumer_key, consumer_secret, access_token, access_token_secret
    if os.path.isfile("apikeys.bin"):
        consumer_key, consumer_secret, access_token, access_token_secret = pickle.load(open("apikeys.bin", "rb")) 
    else:
        consumer_key = input("consumer_key: ")
        consumer_secret = input("consumer_secret: ")
        access_token = input("access_token: ")
        access_token_secret = input("access_token_secret: ")
        pickle.dump((consumer_key, consumer_secret, access_token, access_token_secret), open("apikeys.bin", "wb"))

def twitterAPI():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    return api



def tweet(status, replyto=None, imgfilename=None):
    if not (status or imgfilename):
        return
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', status)
    # take out all url texts from status for count, all urls count as 23
    rstat = status
    for u in urls:
        rstat = rstat.replace(u, '')
    nchars = len(rstat) + 23 * len(urls)
    if nchars > 140:
        print("Tweet too long")
        
    #print(status)
    
    api = twitterAPI()
    if (imgfilename and os.path.isfile(imgfilename)):
        try:
            stat = api.update_with_media(imgfilename, status=status, in_reply_to_status_id=(replyto and replyto.id))
        except Exception as e:
            print(e)
            stat = None
        
    else:
        try:
            stat = api.update_status(status=status, in_reply_to_status_id=(replyto and replyto.id))
        except Exception as e:
            print(e)
            stat = None
    return stat
    
def compose(day, clock, reading):    
    status = ["%s, %s (%s)" % (day, clock, mgm3)]
    skeys = list(reading.keys())
    skeys.sort()
    for k in skeys:
        if reading[k][0] == "n/a":
            status.append("%s: %s" % (k, reading[k][0]))
        else:
            status.append("%s: %.0f %s" % (k, reading[k][0], reading[k][1]))
    status.append("%s" % shorturlstr)
    status = '\n'.join(status)
    return status

def toDT(day, clock):
    if clock[:5] == "24:00": # 27/01/2017 24:00 is in fact 28/01/2017 00:00
        clock = "00:00"
        day = (datetime.strptime(day, "%d/%m/%Y") + timedelta(hours=24)).strftime("%d/%m/%Y")
    return datetime.strptime("%s %s" % (day, clock[:5]), "%d/%m/%Y %H:%M")

def composeAboveTweet(day, clock, above, origtweetstat):
    status = []
    dtnow = toDT(day, clock)
    for k in above:
        # count hours above
        #print("In composeAboveTweet", k, above[k])
        lday, lclock, lvalue = above[k][0]
        if lday == day and lclock == clock:
            stat = []
            # count hours above
            dtlast = dtnow
            nhours = 1
            for lday, lclock, lvalue in above[k][1:]:
                if lday == day and lclock == clock:
                    continue # skip duplicate entries
                dt = toDT(lday, lclock)
                
                if (dtlast - dt) == timedelta(hours=1):
                    nhours += 1
                else:
                    break
                dtlast = dt
            stat.append("@lpoolcouncil @DefraUKAir @LiverpoolFoE: %s %dh above @WHO guide (%.0f%s %s-mean %s) #airpollution #liverpool" % 
                        (k, nhours, guides[k], mgm3, meansWHO[k], urlWHO))
            if meansWHO[k] != meansDEFRA[k]:
                stat.append("(Note #DEFRA data is %s mean)" % meansDEFRA[k])            
            status.append('\n'.join(stat))
    return status
        


def scrape():
    f = request.urlopen(urlstr)

    r = f.read()
    g = re.search(b".*<tr>.*(%s.*?)</tr>" % sitename, r, re.DOTALL)
    #print(g.group(1))

    # split into <td></td>
    row = g.group(1)
    #print("row = %s\n" % row)

    # date and time
    dategroups = re.search(b".*<td>(.*?)<br.*?>(.*?)</td>", row, re.DOTALL)
    day = dategroups.group(1).decode("utf-8")
    clock = dategroups.group(2).decode("utf-8")


    # data
    cols = re.findall(b"<span.*?>(.*?)</span>", row, re.DOTALL)
    assert len(cols) == 5
    units = [O3, NO2, SO2, PM25, PM100]
    datanums = []
    for v in cols:
        value = 'not_set'
        if b' ' in v:
            try:
              value = float(v[:v.index(b' ')])
            except ValueError:
              pass
        if value == 'not_set' and b'n/a' in v:
            value = "n/a"
        else:
            value = float(v[:v.index(b'&')])
        nv = v.replace(b'&nbsp;', b' ')
        ix = b''
        m = re.match(b".*?(\(.*?\))", nv)
        if m:
            ix = re.match(b".*?(\(.*?\))", nv).group(1)
        datanums.append((value, ix.decode("utf-8")))

    reading = dict(zip(units, datanums))
    return day, clock, reading

def convert(r):
    # converts result from sqlite query to format we scrape of day, time, readings
    # where readings is a dict with keys "units" and tuples as values
    units = [O3, NO2, SO2, PM25, PM100]
    converted = deque()
    for e in r:
        dt = datetime.strptime(e[1], "%Y-%m-%d %H:%M:%S.%f")
        date = dt.strftime("%d/%m/%Y")
        clock = dt.strftime("%H:%M:%S")
        tpls = []
        for v in e[2:]:
            if v[:3] == "n/a":
                tpls.append(("n/a", ''))
            else:
                m = re.match("(.*?)(\(.*?\))", v)
                tpls.append((float(m.group(1)), m.group(2)))
        assert len(tpls) == 5
        converted.appendleft((date, clock, dict(zip(units, tpls))))
    return converted



def loadAllReadings(dbname):
    db = sqlite3.connect(dbname)
    c = db.cursor()
    c.execute("SELECT * FROM readings")
    return c.fetchall()

def loadLastReading(dbname):
    db = sqlite3.connect(dbname)
    c = db.cursor()
    c.execute("SELECT * FROM readings WHERE id in ( SELECT max(id) FROM readings)")
    return c.fetchall()

def loadReadings():
    fall = "allreadings.bin"
    allreadings = deque()
    if os.path.isfile(fall):
        allreadings = pickle.load(open(fall, "rb"))
    return allreadings

def saveLastReading(dbname, date, time, reading, overwrt=False):
    units = [O3, NO2, SO2, PM25, PM100]
    db = sqlite3.connect(dbname)
    c = db.cursor()
    if overwrt:
        c.execute(''' DROP TABLE IF EXISTS readings''')
    e = '''
        CREATE TABLE IF NOT EXISTS readings(id INTEGER PRIMARY KEY, date_time TEXT, %s TEXT, %s TEXT, %s TEXT, %s TEXT, %s TEXT)
        ''' % tuple(units)
    c.execute(e)
    dt = datetime.strptime("%s %s" % (date, time), "%d/%m/%Y %H:%M:%S")
    dts = dt.strftime("%Y-%m-%d %H:%M:%S.000")
    c.execute("SELECT * FROM readings WHERE date_time=?", (dts,))
    r = c.fetchall()
    if r:
        print("Already exists")
        return
    e = '''INSERT INTO readings(date_time, %s, %s, %s, %s, %s) VALUES(?,?,?,?,?,?)''' % tuple(units)
    t = (dts, "%s %s" % reading[O3], "%s %s"% reading[NO2], "%s %s" % reading[SO2], "%s %s" % reading[PM25], "%s %s" % reading[PM100])
    c.execute(e, t)
    db.commit()
    db.close()

def pickleReadings(allreading):
    fall = "allreadings.bin"
    pickle.dump(allreadings, open(fall, "wb"))
    
def compareWHO(allreadings):
    above = {}
    for (day, clock, reading) in allreadings:
        for k in guides:
            if type(reading[k][0]) == type(1.0) and reading[k][0] > guides[k]:
                if k not in above:
                    above[k] = []
                above[k].append((day,clock, reading[k][0]))
    return above


def weatherTweetToDict(t):
    m = re.match(".*AirTemp ([\+\-0-9.]*).*?, RH ([0-9]*?)\%, wind speed ([0-9.]*) m\/s, wind dir ([0-9.]*?) deg, Time ([0-9:]*?)UTC", t.text)
    if m:
        try:
            d = {"temp": float(m.group(1)), "rh": int(m.group(2)), "windspeed": float(m.group(3)), "winddir": float(m.group(4)), "time": m.group(5)}
            d["datetime"] = t.created_at
            d["tweet"] = t
            return d
        except Exception as e:
            print(t.text)
            raise e

def getAndPickleWeather(fn, readings):
    api = twitterAPI()
    oldestReading = toDT(readings[-1][0], readings[-1][1])

    idlast = None
    alltweets = []
    while True:
        if 0:#idlast == None:
            r = api.user_timeline("@livuniwx")
        else:
            r = api.user_timeline("@livuniwx", max_id=idlast)
        for i, t in enumerate(r[:-1]):
            d = weatherTweetToDict(t)
            if d:
                alltweets.append(d)

        if r[-1].created_at < oldestReading:
            break
        idlast = r[-1].id
    pickle.dump(alltweets, open(fn, "wb"))
    print("Pickled ", len(alltweets), " tweets")

def loadWeatherTweets(fn):
    wt = pickle.load(open(fn, "rb"))
    d0 = wt[0]["datetime"]
    for t in wt[1:]:
        assert t["datetime"] < d0
        d0 = t["datetime"]
    return wt


def testCMap():
    # colourmap from green over yellow to red
    cdict = {
    'red'  :  ((0.00, 0.00, 0.00), 
               (0.50, 1.00, 1.00),
               (1.00, 1.00, 1.00)),
    'green':  ((0.00, 1.00, 1.00), 
               (0.50, 1.00, 1.00),
               (1.00, 0.00, 0.00)),
    'blue' :  ((0.00, 0.00, 0.00), 
               (0.50, 0.00, 0.00),
               (1.00, 0.00, 0.00)),
    }

    cm = LinearSegmentedColormap("mymap", cdict, 256)


    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))


    fig, axes = plt.subplots(nrows=1)
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    #axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

    axes.imshow(gradient, aspect='auto', cmap=cm)
    pos = list(axes.get_position().bounds)
    x_text = pos[0] - 0.01
    y_text = pos[1] + pos[3]/2.
    #fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

    axes.set_axis_off()

    plt.show()

def Along(lam, a, b):
    return lam * b + (1.0 - lam) * a

class Gauge:
    def __init__(self, dates, data, C):
        self.dates = dates
        self.data = data
        self.C = C
        self.titles = {O3: r"$O_3$", NO2: r"$NO_2$", SO2: r"$SO_2$", PM25: r"$PM_{2.5}$", PM100: r"$PM_{10}$"}
        self.mgpqm = "${\mu gm^{-3}}$"
        self.maxValue = None

        self.fig = plt.figure()  
        self.ax = self.fig.add_subplot(1,1,1) 
        self.ax.set_xlim([-1.2, 1.2])
        self.ax.set_ylim([-0.2, 1.2])
        self.ax.set_aspect("equal")
        plt.axis('off')

        circle = patches.Circle((0, 0), 0.06, color="orange", path_effects=[patheffects.SimplePatchShadow(), patheffects.Normal()])
        circle.zorder = 200
        self.ax.add_artist(circle)
        # 50% available for valmin to valmax, where is limit
        self.valmin = 0
        self.valmax = 1.2 * guides[C]
        self.wholimit = guides[C]
        self.rad = 0.9
        lim = 180.0*(1.0 - self.toDialPos(self.wholimit)[2] / math.pi)
        wedgeBelow = patches.Wedge((0, 0), 1.0, lim, 180.0, color=(0.8, 1, 0.8))
        wedgeAbove = patches.Wedge((0, 0), 1.0, 0.0, lim, color=(1, 0.8, 0.8))
        self.ax.add_patch(wedgeBelow)
        self.ax.add_patch(wedgeAbove)
        
        self.apatch = None
        self.maxArtist = None
        self.lastValue = 0.0
        self.addLabels()
    
    def toDialPos(self, value):
        theta = ((value - self.valmin) / (self.valmax - self.valmin)) * math.pi
        sx, sy = -self.rad * math.cos(theta), self.rad * math.sin(theta)
        return sx, sy, theta

    def drawGauge(self, frame):
        # transform value to angle between 0=valmin and 180=valmax
        value = self.data[frame]
        dialColor = "orange"
        if value == "n/a":
            value = self.lastValue
            dialColor = "grey"
        self.lastValue = value
        sx, sy, theta = self.toDialPos(value)
        if self.apatch:
            self.apatch.remove()
        arrow = patches.FancyArrow(0, 0, sx, sy, color=dialColor, width=0.05, length_includes_head=True, head_width=0.07, path_effects=[patheffects.SimplePatchShadow(), patheffects.Normal()])
        self.apatch = self.ax.add_patch(arrow)
        self.apatch.zorder = 100

        # draw the max value
        if self.maxValue == None or value > self.maxValue:
            rx, ry = -(self.rad+0.07) * math.cos(theta), (self.rad+0.07) * math.sin(theta)
            tx, ty = 0.07 * math.cos(theta), -0.07 * math.sin(theta)
            arrow = patches.FancyArrow(rx, ry, tx, ty, color="red", width=0.0, length_includes_head=True, head_width=0.07, path_effects=[patheffects.SimplePatchShadow(), patheffects.Normal()])
            if self.maxValue != None:
                self.aMaxPatch.remove()
            self.aMaxPatch = self.ax.add_patch(arrow)
            self.maxValue = value
            self.maximTitle = "\n Maximum: %.1f%s, %s" % (self.maxValue, self.mgpqm, self.dates[frame].strftime("%d/%m/%Y %H:%M")) 

        if self.maxArtist:
            self.maxArtist.remove()
        if dialColor == "grey":
            self.ax.set_title(self.titles[self.C] + " %s" % self.dates[frame].strftime("%d/%m/%Y %H:%M"), fontsize=12) 
            self.maxArtist = self.ax.add_artist(Text(0, 1.25 * self.rad, text="No readings recorded!", verticalalignment='baseline', horizontalalignment='center'))
        else:
            self.ax.set_title(self.titles[self.C] + " %s" % self.dates[frame].strftime("%d/%m/%Y %H:%M"), fontsize=12) 
            self.maxArtist = self.ax.add_artist(Text(0, 1.25 * self.rad, text="%s" % (self.maximTitle), verticalalignment='baseline', horizontalalignment='center'))


    def addLabels(self):
        # numbers around the top
        for i in range(11):
            value = Along(i/10.0, self.valmin, self.valmax)
            sx, sy, theta = self.toDialPos(value)
            self.ax.add_artist(Text(sx, sy, text="%.0f" % value, verticalalignment='baseline', horizontalalignment='center', rotation=90.0 - math.degrees(theta)))
        # label what we are showing
        self.ax.add_artist(Text(0, self.rad/2, text="%s\n[%s]" % (self.titles[self.C], self.mgpqm), verticalalignment='baseline', horizontalalignment='center'))
        # WHO guide information
        self.ax.add_artist(Text(0, -0.2 * self.rad, text="WHO Limit: %s%s" % (guides[self.C], self.mgpqm), verticalalignment='baseline', horizontalalignment='center', color=(1, 0.8, 0.8)))

def plotRadial(readings, C):
    dates = [toDT(d, c) for d, c, r in readings]
    data = [r[C][0] for d, c, r in readings] # data

    d0, d1 = dates[0], dates[-1] # date range

    gauge = Gauge(dates, data, C)
    framlist = [0, 0, 0, 0, 0, 0, 0, 0]
    framlist.extend(range(len(data)))
    anim = FuncAnimation(gauge.fig, gauge.drawGauge, frames=framlist, interval=200)
    fn = "gauge_%s.gif" % d1.strftime("%Y%m%d%H%M")
    anim.save(fn, dpi=100, writer='imagemagick')
    plt.close(gauge.fig)
    return fn

def plotLinear(readings, C):
    titles = {O3: r"$O_3$", NO2: r"$NO_2$", SO2: r"$SO_2$", PM25: r"$PM_{2.5}$", PM100: r"$PM_{10}$"}

    dates = [toDT(d, c) for d, c, r in readings]
    data = [r[C][0] for d, c, r in readings] # data

    newdates, newdata = [], []
    for date, val in zip(dates, data):
        if val != 'n/a':
            newdates.append(date)
            newdata.append(val)

    data = newdata
    dates = newdates
    d0, d1 = dates[0], dates[-1] # date range

    fig = plt.figure()                                                               
    ax = fig.add_subplot(1,1,1) 
    

    # format x axis
    ax.xaxis_date()
    ax.set_xlim(d0, d1)
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%Hh'))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    # format y axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d/%m'))
    ax.yaxis.set_major_formatter(FormatStrFormatter(r'%.0f$\frac{\mu g}{m^3}$'))
    ax.set_ylim(0, max(data) + 5)


    # green / red background division above and below WHO guide

    guide = guides[C]
    ax.fill_between([d0, d1, d1, d0], [0, 0, guide, guide], facecolor=(0.8, 1, 0.8), edgecolor="none")
    ax.fill_between([d0, d1, d1, d0], [guide, guide, max(data) + 5, max(data) + 5], facecolor=(1, 0.8, 0.8), edgecolor="none")

    ax.scatter(dates, data)
    ax.set_title(titles[C] + " for %s to %s,\nLiverpool Speke (%s)" % (d0.strftime("%d/%m/%Y"), d1.strftime("%d/%m/%Y"), urlstr), fontsize=10) 
    ax.tick_params(axis='both', which='both', labelsize=10)

    fig.autofmt_xdate()

    plt.grid(which='major')
    fn = "figure_%s.png" % d1.strftime("%Y%m%d")
    plt.savefig(fn, dpi=600)

    return fn


def plotPolar(readings, weathertweets):
    def findInWT(dt, wt):
        for t in wt:
            if t["datetime"] - dt < timedelta(minutes=10):
                return t
        assert 0


    # pair pollution readings with weather data
    pm25 = []
    pm100 = []
    windspeed = []
    winddir = []
    dates = []
    for r in readings:
        d, c, rr = r
        dt = toDT(d, c)
        # find dt in wt
        w = findInWT(dt, weathertweets)
        dates.append(dt)
        if type(rr[PM25][0]) != type(''):
            pm25.append(rr[PM25][0])
            windspeed.append(w["windspeed"])
            winddir.append(w["winddir"])
	#if type(rr[PM100][0]) != type(''):
        #    pm100.append(rr[PM100][0])


    theta = np.radians(winddir)

    # colourmap from green over yellow to red
    cdict = {
    'red'  :  ((0.00, 0.00, 0.00), 
               (0.50, 1.00, 1.00),
               (1.00, 1.00, 1.00)),
    'green':  ((0.00, 1.00, 1.00), 
               (0.50, 1.00, 1.00),
               (1.00, 0.00, 0.00)),
    'blue' :  ((0.00, 0.00, 0.00), 
               (0.50, 0.00, 0.00),
               (1.00, 0.00, 0.00)),
    }
    cm = LinearSegmentedColormap("greentored", cdict, 256)

    ax = plt.subplot(111, projection='polar')
    ax.scatter(theta, windspeed, c=pm25, s=100, cmap=cm, edgecolors='none')
    ax.set_rmax(max(windspeed) + 1)
    ax.set_rticks(np.arange(0, max(windspeed), 1))  # less radial ticks
    ax.set_rlabel_position(300)  # get radial labels away from plotted line
    ax.set_theta_zero_location("S")
    ax.set_theta_direction(-1)
    ax.grid(True)
    # tick locations
    thetaticks = np.arange(0,360,90)
    ax.set_thetagrids(thetaticks, frac=1.01)
    #img = plt.imread("speke.png")
    #plt.imshow(img, extent=[0,10,0,10])

    ax.set_title("PM25 %s to %s" % (allreadings[-1][0], allreadings[0][0]))
    plt.show()


if __name__ == "__main__":
    parser = OptionParser()

    parser = OptionParser(usage='usage: %prog [options] ')
    parser.add_option("-f", "--file", dest="filename",
                      help="", metavar="FILE")
    parser.add_option('-m', '--mode',
                      type='choice',
                      action='store',
                      dest='mode',
                      choices=['plotpollution', 'debug', 'saveweather', 'plotpollutionLinear', 'plotRadial', 'regular'],
                      default='regular',
                      help='Choose mode',)
    (options, args) = parser.parse_args()
    
    
    
    mode = options.mode
    loadAPIKeys()

    #allreadings = loadReadings()
    # remove duplicate entries (could have come in while debugging)
    #ic = 0
    #while ic < len(allreadings):
    #    r = allreadings[ic]
    #    while allreadings.count(r) > 1:
    #        allreadings.remove(r)
    #    ic += 1 


    if mode == 'debug':
        #day, clock, reading = scrape()
        #saveLastReading("readings.db", day, clock, reading)
        #r = loadLastReading("readings.db")
        #c = convert(r)
        #print(c)
        #print(scrape())
        # find when we last posted an image
        files = [f for f in os.listdir('.') if re.match("gauge_[0-9]*.gif", f)]
        if files:
            datelast = max([datetime.strptime(f, "gauge_%Y%m%d%H%M.gif") for f in files])
        else:
            datelast = datetime.today() - timedelta(days=100)
        sincelastplot = (datetime.today() - datelast)
        if (sincelastplot > timedelta(hours=24 * 2)):
            allreadings = convert(loadAllReadings("readings.db"))
            allreadings.reverse()
            readings = [(d, h, r) for (d, h, r) in allreadings if toDT(d, h) >= datelast]
            d0, d1 = toDT(readings[0][0], readings[0][1]), toDT(readings[-1][0], readings[-1][1])
            fn = plotRadial(readings, PM25)
            #tweet(PM25 + "\n%s - %s" % (d0.strftime("%d/%m/%Y"), d1.strftime("%d/%m/%Y")), None, fn)

    elif mode == 'saveweather':
        allreadings = convert(loadAllReadings("readings.db"))
        getAndPickleWeather("weathertweets.bin", allreadings)

    elif mode == 'plotpollution':
        weathertweets = loadWeatherTweets("weathertweets.bin")
        plotPolar(allreadings, weathertweets)

    elif mode == 'plotRadial':
        files = [f for f in os.listdir('.') if re.match("gauge_[0-9]*.gif", f)]
        if files:
            datelast = max([datetime.strptime(f, "gauge_%Y%m%d%H%M.gif") for f in files])
        else:
            datelast = datetime.today() - timedelta(days=100)
        sincelastplot = (datetime.today() - datelast)
        if (sincelastplot > timedelta(hours=24 * 2)):
            allreadings = convert(loadAllReadings("readings.db"))
            allreadings.reverse()
            readings = [(d, h, r) for (d, h, r) in allreadings if toDT(d, h) >= datelast]
            d0, d1 = toDT(readings[0][0], readings[0][1]), toDT(readings[-1][0], readings[-1][1])
            fn = plotRadial(readings, PM25)
            tweet(PM25 + "\n%s - %s" % (d0.strftime("%d/%m/%Y"), d1.strftime("%d/%m/%Y")), None, fn)

    elif mode == "plotpollutionLinear":
        # find when we last posted an image
        files = [f for f in os.listdir('.') if re.match("figure_[0-9]*.png", f)]
        if files:
            datelast = max([datetime.strptime(f, "figure_%Y%m%d.png") for f in files])
            datelast += timedelta(hours=12)
        else:
            datelast = datetime.today() - timedelta(days=100)
        sincelastplot = (datetime.today() - datelast)
        if (sincelastplot > timedelta(hours=24 * 3)):
            allreadings = convert(loadAllReadings("readings.db"))
            allreadings.reverse()
            readings = [(d, h, r) for (d, h, r) in allreadings if toDT(d, h) >= datelast]
            figure = plotLinear(readings, PM25)
            d0, d1 = toDT(readings[0][0], readings[0][1]), toDT(readings[-1][0], readings[-1][1])
            #tweet(PM25 + "\n%s - %s" % (d0.strftime("%d/%m/%Y"), d1.strftime("%d/%m/%Y")), None, figure)


    else:
        day, clock, reading = scrape()
        r = loadLastReading("readings.db")
        converted = convert(r)
        assert(len(converted) == 1)
        lastday, lastclock, lastreading = converted[-1]
        if ((day, clock) != (lastday, lastclock)):
            status = compose(day, clock, reading)
            rtweet = tweet(status)

            saveLastReading("readings.db", day, clock, reading)
            allreadings = convert(loadAllReadings("readings.db"))


            # compare with WHO recommendations
            r = allreadings and compareWHO(allreadings)
            if r:
                stats = composeAboveTweet(day, clock, r, rtweet)
                for s in stats:
                    tweet(s, replyto=rtweet)
        else:
            print("Reading already known")

