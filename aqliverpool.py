#!/usr/bin/python3

# coding: utf-8

from optparse import OptionParser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import urllib
from urllib import request
import re
import html
import sys, os
import pickle
from datetime import datetime, timedelta
import tweepy

from collections import deque

urlstr = "https://uk-air.defra.gov.uk/latest/currentlevels?view=site#L"
shorturlstr = "https://goo.gl/ZpELjS"

urlWHO = "http://apps.who.int/iris/bitstream/10665/69477/1/WHO_SDE_PHE_OEH_06.02_eng.pdf"

sitename = b'Liverpool'

mgm3 = '\u03BCgm\u207B\u00B3'
O3, NO2, SO2, PM25, PM100 = "O\u2083", "NO\u2082", "SO\u2082", "PM\u2082\u2085", "PM\u2081\u2080\u2080"
guides = {O3:100, NO2:200, SO2:20, PM25:25, PM100:50} # source: http://apps.who.int/iris/bitstream/10665/69477/1/WHO_SDE_PHE_OEH_06.02_eng.pdf  
meansWHO = {O3:'8h', NO2:'1h', SO2:'10m', PM25:'24h', PM100:'24h'}
meansDEFRA = {O3:'8h', NO2:'1h', SO2:'max 15m', PM25:'24h', PM100:'24h'}

consumer_key, consumer_secret, access_token, access_token_secret = pickle.load(open("apikeys.bin", "rb")) 

def twitterAPI():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    return api



def tweet(status, replyto=None, imgfilename=None):
    if not (status or imgfilename):
        return
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', status)
    print("urls = ", urls)
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
        print(v)
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

def loadReadings():
    fall = "allreadings.bin"
    allreadings = deque()
    if os.path.isfile(fall):
        allreadings = pickle.load(open(fall, "rb"))
    return allreadings

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
    #wt = pickle.load(open("weathertweets.bin", "rb"))
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
                      choices=['plotpollution', 'debug', 'saveweather', 'plotpollutionLinear', 'regular'],
                      default='regular',
                      help='Choose mode',)
    (options, args) = parser.parse_args()
    
    
    
    mode = options.mode

    allreadings = loadReadings()
    # remove duplicate entries (could have come in while debugging)
    ic = 0
    while ic < len(allreadings):
        r = allreadings[ic]
        while allreadings.count(r) > 1:
            allreadings.remove(r)
        ic += 1 


    if mode == 'debug':
        stat = tweet("TTEESSTT")
        print(stat)
        #tweet("In reply to: TEST3", stat)

    elif mode == 'saveweather':
        allreadings = loadReadings()
        getAndPickleWeather("weathertweets.bin", allreadings)

    elif mode == 'plotpollution':
        weathertweets = loadWeatherTweets("weathertweets.bin")
        plotPolar(allreadings, weathertweets)

    elif mode == "plotpollutionLinear":
        # find when we last posted an image
        files = [f for f in os.listdir('.') if re.match("figure_[0-9]*.png", f)]
        if files:
            datelast = max([datetime.strptime(f, "figure_%Y%m%d.png") for f in files])
            datelast += timedelta(hours=12)
        else:
            datelast = datetime.today() - timedelta(days=100)
        print(datelast)
        sincelastplot = (datetime.today() - datelast)
        if (sincelastplot > timedelta(hours=24 * 3)):
            allreadings.reverse()
            readings = [(d, h, r) for (d, h, r) in allreadings if toDT(d, h) >= datelast]
            figure = plotLinear(readings, PM25)
            d0, d1 = toDT(readings[0][0], readings[0][1]), toDT(readings[-1][0], readings[-1][1])
            tweet(PM25 + "\n%s - %s" % (d0.strftime("%d/%m/%Y"), d1.strftime("%d/%m/%Y")), None, figure)


    else:
        if allreadings:
            lastday, lastclock, lastreading = allreadings[0]
        else:
            lastday, lastclock, lastreadings = None, None, None
        day, clock, reading = scrape()
        if ((day, clock) != (lastday, lastclock)):
            status = compose(day, clock, reading)
            rtweet = tweet(status)

            allreadings.appendleft((day, clock, reading))
            pickleReadings(allreadings)

            # compare with WHO recommendations
            r = allreadings and compareWHO(allreadings)
            if r:
                stats = composeAboveTweet(day, clock, r, rtweet)
                for s in stats:
                    tweet(s, replyto=rtweet)
        else:
            print("Reading already known")

