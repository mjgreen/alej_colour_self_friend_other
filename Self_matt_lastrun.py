#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on October 08, 2025, at 15:48
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'Self_matt'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'block order': ['color first','identity first'],
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = False
_winSize = [1280, 600]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Matt\\gits\\alej_colour_self_friend_other\\Self_matt_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='testMonitor', color=[1.0000, 1.0000, 1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [1.0000, 1.0000, 1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp_3') is None:
        # initialise key_resp_3
        key_resp_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_3',
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    if deviceManager.getDevice('key_resp_instrucctions') is None:
        # initialise key_resp_instrucctions
        key_resp_instrucctions = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_instrucctions',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "preliminaries" ---
    # Run 'Begin Experiment' code from assign_keymap_at_random_and_mk_instructions
    # choose one of 6 combinations of j,k,l mapping to redself; bluefriend; othergreen
    #redself   bluefriend  greenother
    #1=j              k                    l
    #2=j              l                     k
    #3=k             j                     l
    #4=k             l                     j
    #5=l              j                     k
    #6=l              k                    j
    
    import random
    mapping_number = random.randint(1,6)
    match mapping_number:
        case 1:
            redself    = "j"
            bluefriend = "k"
            greenother = "l"
        case 2:
            redself    = "j"
            bluefriend = "l"
            greenother = "k"
        case 3:
            redself    = "k"
            bluefriend = "j"
            greenother = "l"
        case 4:
            redself    = "k"
            bluefriend = "l"
            greenother = "j"
        case 5:
            redself    = "l"
            bluefriend = "j"
            greenother = "k"
        case 6:
            redself    = "l"
            bluefriend = "k"
            greenother = "j"                        
        case _:
            print("oops")
    
    # create strings from keymap for use in instructions
    #e.g., instr_1 = "press" + keymap1key1 + "for redself"
    instruct_color_condition = "In the following task, you will need to report the color of the word. \n- If the word is presented in blue, press " + bluefriend + "\n- If the word is presented in red, press " + redself + "\n- If the word is presented in green, press " + greenother + "\n\npress the space key to start"
    instruct_identity_condition = "In the following task, you will need to report the identity of the word.\n- If the word is self, press " + redself + "\n- If the word is friend, press " + bluefriend + "\n- If the word is other, press " + greenother + "\n\npress the space key to start"
    # Run 'Begin Experiment' code from pull_fonts_in
    font_files = ['Arial.ttf', 'verdana.ttf', 'tahoma.ttf', 'RAVIE.ttf', 'JOKERMAN.ttf']
    font_main1 = 'Arial'
    font_main2 = 'verdana'
    font_main3 = 'tahoma'
    font_main4 = 'RAVIE'
    font_main5 = 'JOKERMAN'
    
    _ = visual.TextStim(win=win, font=font_main1, fontFiles=font_files)
    del _
    
    _ = visual.TextStim(win=win, font=font_main2, fontFiles=font_files)
    del _
    
    _ = visual.TextStim(win=win, font=font_main3, fontFiles=font_files)
    del _
    
    _ = visual.TextStim(win=win, font=font_main4, fontFiles=font_files)
    del _
    
    _ = visual.TextStim(win=win, font=font_main5, fontFiles=font_files)
    del _
    
    # --- Initialize components for Routine "keymap_test" ---
    text = visual.TextStim(win=win, name='text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=1.5, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_3 = keyboard.Keyboard(deviceName='key_resp_3')
    
    # --- Initialize components for Routine "font_selection_test" ---
    text_title = visual.TextStim(win=win, name='text_title',
        text='Demonstration that font selection works - remove for real runs',
        font='Arial',
        pos=(0, 0.4), draggable=False, height=0.1, wrapWidth=1.5, ori=0.0, 
        color='blue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text1_arial = visual.TextStim(win=win, name='text1_arial',
        text='I am Arial abcdefghijklmnopqrstuvwxyz123',
        font='Arial',
        pos=(0, 0.1), draggable=False, height=0.05, wrapWidth=1.5, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text1_verdana = visual.TextStim(win=win, name='text1_verdana',
        text='I am Verdana abcdefghijklmnopqrstuvwxyz123',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=1.5, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text1_tahoma = visual.TextStim(win=win, name='text1_tahoma',
        text='I am Tahoma abcdefghijklmnopqrstuvwxyz123',
        font='Arial',
        pos=(0, -.1), draggable=False, height=0.05, wrapWidth=1.5, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    text1_ravie = visual.TextStim(win=win, name='text1_ravie',
        text='I am Ravie abcdefghijklmnopqrstuvwxyz123',
        font='Arial',
        pos=(0, -.2), draggable=False, height=0.05, wrapWidth=1.5, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    text1_jokerman = visual.TextStim(win=win, name='text1_jokerman',
        text='I am Jokerman abcdefghijklmnopqrstuvwxyz123',
        font='Arial',
        pos=(0, -0.3), draggable=False, height=0.05, wrapWidth=1.5, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    text_dismiss = visual.TextStim(win=win, name='text_dismiss',
        text='Press space to continue',
        font='Arial',
        pos=(0, -0.45), draggable=False, height=0.05, wrapWidth=1.5, ori=0.0, 
        color='blue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    
    # --- Initialize components for Routine "instructions" ---
    text_instructions = visual.TextStim(win=win, name='text_instructions',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_instrucctions = keyboard.Keyboard(deviceName='key_resp_instrucctions')
    
    # --- Initialize components for Routine "say_stuff" ---
    
    # --- Initialize components for Routine "stimulus" ---
    # Run 'Begin Experiment' code from fix_duration_code
    # Fixation cross duration
    fixDurMin = 0.3 # minimum: 300 ms
    fixDurMax = 1 # maximum: 700 ms
    fix = visual.TextStim(win=win, name='fix',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    stimuli = visual.TextStim(win=win, name='stimuli',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.25, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "blank" ---
    blank_screen = visual.TextStim(win=win, name='blank_screen',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "increment_foocount" ---
    
    # --- Initialize components for Routine "the_end" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='The End',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "preliminaries" ---
    # create an object to store info about Routine preliminaries
    preliminaries = data.Routine(
        name='preliminaries',
        components=[],
    )
    preliminaries.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from parse_block_order_from_gui
    foo=['bar','daz']
    print("first block is: ", expInfo['block order'])
    if expInfo['block order'] == "color first":
        foo=['trials_color.xlsx','trials_identity.xlsx']
    if expInfo['block order'] == "identity first":
        foo=['trials_identity.xlsx','trials_color.xlsx']
    foocount=0
    condition=foo[foocount]
    print("condition is: ",condition)
    
    # store start times for preliminaries
    preliminaries.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    preliminaries.tStart = globalClock.getTime(format='float')
    preliminaries.status = STARTED
    thisExp.addData('preliminaries.started', preliminaries.tStart)
    preliminaries.maxDuration = None
    # keep track of which components have finished
    preliminariesComponents = preliminaries.components
    for thisComponent in preliminaries.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "preliminaries" ---
    preliminaries.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=preliminaries,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            preliminaries.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in preliminaries.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "preliminaries" ---
    for thisComponent in preliminaries.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for preliminaries
    preliminaries.tStop = globalClock.getTime(format='float')
    preliminaries.tStopRefresh = tThisFlipGlobal
    thisExp.addData('preliminaries.stopped', preliminaries.tStop)
    thisExp.nextEntry()
    # the Routine "preliminaries" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "keymap_test" ---
    # create an object to store info about Routine keymap_test
    keymap_test = data.Routine(
        name='keymap_test',
        components=[text, key_resp_3],
    )
    keymap_test.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_5
    debug_string = "Demonstration that random selection of keymap works - remove for real runs\n\nRandomly selected mapping_number is " + str(mapping_number) + " of 6" + ";\nredself key is " + redself + ";\nbluefriend key is " + bluefriend + ";\ngreenother key is " + greenother + "\n\nPress any key to continue"
    text.setText(debug_string)
    # create starting attributes for key_resp_3
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # store start times for keymap_test
    keymap_test.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    keymap_test.tStart = globalClock.getTime(format='float')
    keymap_test.status = STARTED
    thisExp.addData('keymap_test.started', keymap_test.tStart)
    keymap_test.maxDuration = None
    # keep track of which components have finished
    keymap_testComponents = keymap_test.components
    for thisComponent in keymap_test.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "keymap_test" ---
    keymap_test.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *key_resp_3* updates
        waitOnFlip = False
        
        # if key_resp_3 is starting this frame...
        if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_3.started')
            # update status
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_3.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=keymap_test,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            keymap_test.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in keymap_test.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "keymap_test" ---
    for thisComponent in keymap_test.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for keymap_test
    keymap_test.tStop = globalClock.getTime(format='float')
    keymap_test.tStopRefresh = tThisFlipGlobal
    thisExp.addData('keymap_test.stopped', keymap_test.tStop)
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    thisExp.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        thisExp.addData('key_resp_3.rt', key_resp_3.rt)
        thisExp.addData('key_resp_3.duration', key_resp_3.duration)
    thisExp.nextEntry()
    # the Routine "keymap_test" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "font_selection_test" ---
    # create an object to store info about Routine font_selection_test
    font_selection_test = data.Routine(
        name='font_selection_test',
        components=[text_title, text1_arial, text1_verdana, text1_tahoma, text1_ravie, text1_jokerman, text_dismiss, key_resp_2],
    )
    font_selection_test.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    text1_arial.setFont(font_main1)
    text1_verdana.setFont(font_main2)
    text1_tahoma.setFont(font_main3)
    text1_ravie.setFont(font_main4)
    text1_jokerman.setFont(font_main5)
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # store start times for font_selection_test
    font_selection_test.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    font_selection_test.tStart = globalClock.getTime(format='float')
    font_selection_test.status = STARTED
    thisExp.addData('font_selection_test.started', font_selection_test.tStart)
    font_selection_test.maxDuration = None
    # keep track of which components have finished
    font_selection_testComponents = font_selection_test.components
    for thisComponent in font_selection_test.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "font_selection_test" ---
    font_selection_test.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_title* updates
        
        # if text_title is starting this frame...
        if text_title.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_title.frameNStart = frameN  # exact frame index
            text_title.tStart = t  # local t and not account for scr refresh
            text_title.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_title, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_title.started')
            # update status
            text_title.status = STARTED
            text_title.setAutoDraw(True)
        
        # if text_title is active this frame...
        if text_title.status == STARTED:
            # update params
            pass
        
        # *text1_arial* updates
        
        # if text1_arial is starting this frame...
        if text1_arial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text1_arial.frameNStart = frameN  # exact frame index
            text1_arial.tStart = t  # local t and not account for scr refresh
            text1_arial.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text1_arial, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text1_arial.started')
            # update status
            text1_arial.status = STARTED
            text1_arial.setAutoDraw(True)
        
        # if text1_arial is active this frame...
        if text1_arial.status == STARTED:
            # update params
            pass
        
        # *text1_verdana* updates
        
        # if text1_verdana is starting this frame...
        if text1_verdana.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text1_verdana.frameNStart = frameN  # exact frame index
            text1_verdana.tStart = t  # local t and not account for scr refresh
            text1_verdana.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text1_verdana, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text1_verdana.started')
            # update status
            text1_verdana.status = STARTED
            text1_verdana.setAutoDraw(True)
        
        # if text1_verdana is active this frame...
        if text1_verdana.status == STARTED:
            # update params
            pass
        
        # *text1_tahoma* updates
        
        # if text1_tahoma is starting this frame...
        if text1_tahoma.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text1_tahoma.frameNStart = frameN  # exact frame index
            text1_tahoma.tStart = t  # local t and not account for scr refresh
            text1_tahoma.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text1_tahoma, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text1_tahoma.started')
            # update status
            text1_tahoma.status = STARTED
            text1_tahoma.setAutoDraw(True)
        
        # if text1_tahoma is active this frame...
        if text1_tahoma.status == STARTED:
            # update params
            pass
        
        # *text1_ravie* updates
        
        # if text1_ravie is starting this frame...
        if text1_ravie.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text1_ravie.frameNStart = frameN  # exact frame index
            text1_ravie.tStart = t  # local t and not account for scr refresh
            text1_ravie.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text1_ravie, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text1_ravie.started')
            # update status
            text1_ravie.status = STARTED
            text1_ravie.setAutoDraw(True)
        
        # if text1_ravie is active this frame...
        if text1_ravie.status == STARTED:
            # update params
            pass
        
        # *text1_jokerman* updates
        
        # if text1_jokerman is starting this frame...
        if text1_jokerman.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text1_jokerman.frameNStart = frameN  # exact frame index
            text1_jokerman.tStart = t  # local t and not account for scr refresh
            text1_jokerman.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text1_jokerman, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text1_jokerman.started')
            # update status
            text1_jokerman.status = STARTED
            text1_jokerman.setAutoDraw(True)
        
        # if text1_jokerman is active this frame...
        if text1_jokerman.status == STARTED:
            # update params
            pass
        
        # *text_dismiss* updates
        
        # if text_dismiss is starting this frame...
        if text_dismiss.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_dismiss.frameNStart = frameN  # exact frame index
            text_dismiss.tStart = t  # local t and not account for scr refresh
            text_dismiss.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_dismiss, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_dismiss.started')
            # update status
            text_dismiss.status = STARTED
            text_dismiss.setAutoDraw(True)
        
        # if text_dismiss is active this frame...
        if text_dismiss.status == STARTED:
            # update params
            pass
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=font_selection_test,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            font_selection_test.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in font_selection_test.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "font_selection_test" ---
    for thisComponent in font_selection_test.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for font_selection_test
    font_selection_test.tStop = globalClock.getTime(format='float')
    font_selection_test.tStopRefresh = tThisFlipGlobal
    thisExp.addData('font_selection_test.stopped', font_selection_test.tStop)
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "font_selection_test" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    blocks = data.TrialHandler2(
        name='blocks',
        nReps=2.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(blocks)  # add the loop to the experiment
    thisBlock = blocks.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            globals()[paramName] = thisBlock[paramName]
    
    for thisBlock in blocks:
        blocks.status = STARTED
        if hasattr(thisBlock, 'status'):
            thisBlock.status = STARTED
        currentLoop = blocks
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                globals()[paramName] = thisBlock[paramName]
        
        # --- Prepare to start Routine "instructions" ---
        # create an object to store info about Routine instructions
        instructions = data.Routine(
            name='instructions',
            components=[text_instructions, key_resp_instrucctions],
        )
        instructions.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from assign_instruction_text
        if condition == 'trials_color.xlsx':
            instruction = instruct_color_condition
        if condition == 'trials_identity.xlsx':
            instruction = instruct_identity_condition
            
        text_instructions.setText(instruction)
        # create starting attributes for key_resp_instrucctions
        key_resp_instrucctions.keys = []
        key_resp_instrucctions.rt = []
        _key_resp_instrucctions_allKeys = []
        # store start times for instructions
        instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        instructions.tStart = globalClock.getTime(format='float')
        instructions.status = STARTED
        thisExp.addData('instructions.started', instructions.tStart)
        instructions.maxDuration = None
        # keep track of which components have finished
        instructionsComponents = instructions.components
        for thisComponent in instructions.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "instructions" ---
        instructions.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlock, 'status') and thisBlock.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_instructions* updates
            
            # if text_instructions is starting this frame...
            if text_instructions.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                text_instructions.frameNStart = frameN  # exact frame index
                text_instructions.tStart = t  # local t and not account for scr refresh
                text_instructions.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_instructions, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_instructions.started')
                # update status
                text_instructions.status = STARTED
                text_instructions.setAutoDraw(True)
            
            # if text_instructions is active this frame...
            if text_instructions.status == STARTED:
                # update params
                pass
            
            # *key_resp_instrucctions* updates
            waitOnFlip = False
            
            # if key_resp_instrucctions is starting this frame...
            if key_resp_instrucctions.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_instrucctions.frameNStart = frameN  # exact frame index
                key_resp_instrucctions.tStart = t  # local t and not account for scr refresh
                key_resp_instrucctions.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_instrucctions, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_instrucctions.started')
                # update status
                key_resp_instrucctions.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_instrucctions.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_instrucctions.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_instrucctions.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_instrucctions.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_instrucctions_allKeys.extend(theseKeys)
                if len(_key_resp_instrucctions_allKeys):
                    key_resp_instrucctions.keys = _key_resp_instrucctions_allKeys[-1].name  # just the last key pressed
                    key_resp_instrucctions.rt = _key_resp_instrucctions_allKeys[-1].rt
                    key_resp_instrucctions.duration = _key_resp_instrucctions_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=instructions,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                instructions.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in instructions.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "instructions" ---
        for thisComponent in instructions.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for instructions
        instructions.tStop = globalClock.getTime(format='float')
        instructions.tStopRefresh = tThisFlipGlobal
        thisExp.addData('instructions.stopped', instructions.tStop)
        # check responses
        if key_resp_instrucctions.keys in ['', [], None]:  # No response was made
            key_resp_instrucctions.keys = None
        blocks.addData('key_resp_instrucctions.keys',key_resp_instrucctions.keys)
        if key_resp_instrucctions.keys != None:  # we had a response
            blocks.addData('key_resp_instrucctions.rt', key_resp_instrucctions.rt)
            blocks.addData('key_resp_instrucctions.duration', key_resp_instrucctions.duration)
        # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler2(
            name='trials',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(condition), 
            seed=None, 
        )
        thisExp.addLoop(trials)  # add the loop to the experiment
        thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrial in trials:
            trials.status = STARTED
            if hasattr(thisTrial, 'status'):
                thisTrial.status = STARTED
            currentLoop = trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "say_stuff" ---
            # create an object to store info about Routine say_stuff
            say_stuff = data.Routine(
                name='say_stuff',
                components=[],
            )
            say_stuff.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_3
            print("hello")
            # store start times for say_stuff
            say_stuff.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            say_stuff.tStart = globalClock.getTime(format='float')
            say_stuff.status = STARTED
            thisExp.addData('say_stuff.started', say_stuff.tStart)
            say_stuff.maxDuration = None
            # keep track of which components have finished
            say_stuffComponents = say_stuff.components
            for thisComponent in say_stuff.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "say_stuff" ---
            say_stuff.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=say_stuff,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    say_stuff.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in say_stuff.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "say_stuff" ---
            for thisComponent in say_stuff.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for say_stuff
            say_stuff.tStop = globalClock.getTime(format='float')
            say_stuff.tStopRefresh = tThisFlipGlobal
            thisExp.addData('say_stuff.stopped', say_stuff.tStop)
            # the Routine "say_stuff" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "stimulus" ---
            # create an object to store info about Routine stimulus
            stimulus = data.Routine(
                name='stimulus',
                components=[fix, stimuli, key_resp],
            )
            stimulus.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from fix_duration_code
            fixDur = np.random.uniform(fixDurMin, fixDurMax) # random between minimum and maximum
            thisExp.addData('fixation_duration',fixDur) # document the duration in your logfile
            stimuli.setColor(color, colorSpace='rgb')
            stimuli.setText(word)
            stimuli.setFont(font)
            # create starting attributes for key_resp
            key_resp.keys = []
            key_resp.rt = []
            _key_resp_allKeys = []
            # store start times for stimulus
            stimulus.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            stimulus.tStart = globalClock.getTime(format='float')
            stimulus.status = STARTED
            thisExp.addData('stimulus.started', stimulus.tStart)
            stimulus.maxDuration = None
            # keep track of which components have finished
            stimulusComponents = stimulus.components
            for thisComponent in stimulus.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "stimulus" ---
            stimulus.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *fix* updates
                
                # if fix is starting this frame...
                if fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fix.frameNStart = frameN  # exact frame index
                    fix.tStart = t  # local t and not account for scr refresh
                    fix.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fix, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fix.started')
                    # update status
                    fix.status = STARTED
                    fix.setAutoDraw(True)
                
                # if fix is active this frame...
                if fix.status == STARTED:
                    # update params
                    pass
                
                # if fix is stopping this frame...
                if fix.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fix.tStartRefresh + fixDur-frameTolerance:
                        # keep track of stop time/frame for later
                        fix.tStop = t  # not accounting for scr refresh
                        fix.tStopRefresh = tThisFlipGlobal  # on global time
                        fix.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fix.stopped')
                        # update status
                        fix.status = FINISHED
                        fix.setAutoDraw(False)
                
                # *stimuli* updates
                
                # if stimuli is starting this frame...
                if stimuli.status == NOT_STARTED and fix.status==FINISHED:
                    # keep track of start time/frame for later
                    stimuli.frameNStart = frameN  # exact frame index
                    stimuli.tStart = t  # local t and not account for scr refresh
                    stimuli.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(stimuli, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'stimuli.started')
                    # update status
                    stimuli.status = STARTED
                    stimuli.setAutoDraw(True)
                
                # if stimuli is active this frame...
                if stimuli.status == STARTED:
                    # update params
                    pass
                
                # *key_resp* updates
                waitOnFlip = False
                
                # if key_resp is starting this frame...
                if key_resp.status == NOT_STARTED and fix.status==FINISHED:
                    # keep track of start time/frame for later
                    key_resp.frameNStart = frameN  # exact frame index
                    key_resp.tStart = t  # local t and not account for scr refresh
                    key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.started')
                    # update status
                    key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp.getKeys(keyList=['j', 'k', 'l'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_allKeys.extend(theseKeys)
                    if len(_key_resp_allKeys):
                        key_resp.keys = _key_resp_allKeys[0].name  # just the first key pressed
                        key_resp.rt = _key_resp_allKeys[0].rt
                        key_resp.duration = _key_resp_allKeys[0].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=stimulus,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    stimulus.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in stimulus.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "stimulus" ---
            for thisComponent in stimulus.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for stimulus
            stimulus.tStop = globalClock.getTime(format='float')
            stimulus.tStopRefresh = tThisFlipGlobal
            thisExp.addData('stimulus.stopped', stimulus.tStop)
            # check responses
            if key_resp.keys in ['', [], None]:  # No response was made
                key_resp.keys = None
            trials.addData('key_resp.keys',key_resp.keys)
            if key_resp.keys != None:  # we had a response
                trials.addData('key_resp.rt', key_resp.rt)
                trials.addData('key_resp.duration', key_resp.duration)
            # the Routine "stimulus" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "blank" ---
            # create an object to store info about Routine blank
            blank = data.Routine(
                name='blank',
                components=[blank_screen],
            )
            blank.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for blank
            blank.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            blank.tStart = globalClock.getTime(format='float')
            blank.status = STARTED
            thisExp.addData('blank.started', blank.tStart)
            blank.maxDuration = None
            # keep track of which components have finished
            blankComponents = blank.components
            for thisComponent in blank.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "blank" ---
            blank.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # if trial has changed, end Routine now
                if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *blank_screen* updates
                
                # if blank_screen is starting this frame...
                if blank_screen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    blank_screen.frameNStart = frameN  # exact frame index
                    blank_screen.tStart = t  # local t and not account for scr refresh
                    blank_screen.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(blank_screen, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blank_screen.started')
                    # update status
                    blank_screen.status = STARTED
                    blank_screen.setAutoDraw(True)
                
                # if blank_screen is active this frame...
                if blank_screen.status == STARTED:
                    # update params
                    pass
                
                # if blank_screen is stopping this frame...
                if blank_screen.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > blank_screen.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        blank_screen.tStop = t  # not accounting for scr refresh
                        blank_screen.tStopRefresh = tThisFlipGlobal  # on global time
                        blank_screen.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'blank_screen.stopped')
                        # update status
                        blank_screen.status = FINISHED
                        blank_screen.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=blank,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    blank.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in blank.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "blank" ---
            for thisComponent in blank.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for blank
            blank.tStop = globalClock.getTime(format='float')
            blank.tStopRefresh = tThisFlipGlobal
            thisExp.addData('blank.stopped', blank.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if blank.maxDurationReached:
                routineTimer.addTime(-blank.maxDuration)
            elif blank.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            # mark thisTrial as finished
            if hasattr(thisTrial, 'status'):
                thisTrial.status = FINISHED
            # if awaiting a pause, pause now
            if trials.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                trials.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'trials'
        trials.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "increment_foocount" ---
        # create an object to store info about Routine increment_foocount
        increment_foocount = data.Routine(
            name='increment_foocount',
            components=[],
        )
        increment_foocount.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code
        if foocount<1:
            foocount+=1
            condition=foo[foocount]
            print("condition is: ",condition)
        # store start times for increment_foocount
        increment_foocount.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        increment_foocount.tStart = globalClock.getTime(format='float')
        increment_foocount.status = STARTED
        thisExp.addData('increment_foocount.started', increment_foocount.tStart)
        increment_foocount.maxDuration = None
        # keep track of which components have finished
        increment_foocountComponents = increment_foocount.components
        for thisComponent in increment_foocount.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "increment_foocount" ---
        increment_foocount.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlock, 'status') and thisBlock.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=increment_foocount,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                increment_foocount.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in increment_foocount.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "increment_foocount" ---
        for thisComponent in increment_foocount.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for increment_foocount
        increment_foocount.tStop = globalClock.getTime(format='float')
        increment_foocount.tStopRefresh = tThisFlipGlobal
        thisExp.addData('increment_foocount.stopped', increment_foocount.tStop)
        # the Routine "increment_foocount" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisBlock as finished
        if hasattr(thisBlock, 'status'):
            thisBlock.status = FINISHED
        # if awaiting a pause, pause now
        if blocks.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            blocks.status = STARTED
    # completed 2.0 repeats of 'blocks'
    blocks.status = FINISHED
    
    
    # --- Prepare to start Routine "the_end" ---
    # create an object to store info about Routine the_end
    the_end = data.Routine(
        name='the_end',
        components=[text_2],
    )
    the_end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_4
    print("The End")
    # store start times for the_end
    the_end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    the_end.tStart = globalClock.getTime(format='float')
    the_end.status = STARTED
    thisExp.addData('the_end.started', the_end.tStart)
    the_end.maxDuration = None
    # keep track of which components have finished
    the_endComponents = the_end.components
    for thisComponent in the_end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "the_end" ---
    the_end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_2.started')
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        
        # if text_2 is stopping this frame...
        if text_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_2.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text_2.tStop = t  # not accounting for scr refresh
                text_2.tStopRefresh = tThisFlipGlobal  # on global time
                text_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.stopped')
                # update status
                text_2.status = FINISHED
                text_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=the_end,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            the_end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in the_end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "the_end" ---
    for thisComponent in the_end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for the_end
    the_end.tStop = globalClock.getTime(format='float')
    the_end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('the_end.stopped', the_end.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if the_end.maxDurationReached:
        routineTimer.addTime(-the_end.maxDuration)
    elif the_end.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
