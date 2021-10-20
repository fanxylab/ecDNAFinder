#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import traceback

from .EcLogging import Logger
from .EcArguments import Args
from .EcPipeline import EcDNA

info ='''
***********************************************************
* Author : Zhou Wei                                       *
* Date   : %s                       *
* E-mail : welljoea@gmail.com                             *
* You are using The scripted by Zhou Wei.                 *
* If you find some bugs, please email to me.              *
* Please let me know and acknowledge in your publication. *
* Sincerely                                               *
* Best wishes!                                            *
***********************************************************
'''%(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))

def Commands():
    args = Args()
    os.makedirs(args.outdir, exist_ok=True)
    Log = Logger( '%s/%s_log.log'%(args.outdir, args.commands) )

    Log.NI(info.strip())
    Log.NI("The argument you have set as follows:".center(59, '*'))

    _ml = max([ len(i) for i in vars(args).keys()] )
    for i,(k,v) in enumerate(vars(args).items(), start=1):
        Log.NI( '**{:0>{}}|{:<{}}: {}'.format(i,2, k,_ml, v) )
    Log.NI(59 * '*')

    try:
        EcDNA(args, Log).Pipeline()
        Log.CI('Success!!!')
    except Exception:
        Log.CW('Failed!!!')
        traceback.print_exc()
    finally:
        Log.CI('You can check your progress in log file.')