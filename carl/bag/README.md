to run cv:

  mkdir h5file cv sub
  
  python write_script.py
  
  sh run.sh
  
  python blendxgbcv.py
  
  #don't run this, bad performance, python blendcv.py 

to rub submission:

  python write_script_sub.py
  
  sh runsub.sh
  
  python xgbcombine.py
  
to run stacking:

  python stackcv.py
  
  python stacksub.py
