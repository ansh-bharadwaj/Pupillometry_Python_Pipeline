# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 22:48:43 2023

@author: anshbharadwaj
"""
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
dataframe_pupil_series = pd.read_excel('C:/Users/ansha/Downloads/sampleEyeData.xlsx', header=None, sheet_name='Pupil')
dataframe_FixationTimeStamp=pd.read_excel('C:/Users/ansha/Downloads/sampleEyeData.xlsx', header=None, sheet_name='FixOn')
dataframe_RewardCueOn=pd.read_excel('C:/Users/ansha/Downloads/sampleEyeData.xlsx', header=None, sheet_name='RwdCueON')
dataframe_RewardCue=pd.read_excel('C:/Users/ansha/Downloads/sampleEyeData.xlsx',header=None,sheet_name="RwdCue")

#self_reward_series=dataframe_pupil_series.iloc[[0,1,2,5,6,9,14,19,21,22,23,25,27]]
#other_reward_series=dataframe_pupil_series.iloc[[3,4,7,8,10,11,12,13,15,16,17,18,20,24,26,28,29]]
#print(self_reward_series)
self_reward_data_series=dataframe_pupil_series.iloc[2]
other_reward_data_series=dataframe_pupil_series.iloc[4]
self_reward_data_series=self_reward_data_series[~np.isnan(self_reward_data_series)]
other_reward_data_series=other_reward_data_series[~np.isnan(other_reward_data_series)]

"""getting 30 trials"""

baseline_avg_array=[]
for big_loop in range(30):
    working_pupil_data=dataframe_pupil_series.iloc[big_loop]
   
    working_pupil_data=working_pupil_data[~np.isnan(working_pupil_data)]
    
    
    print(working_pupil_data)
    working_RewardCueTime_data=dataframe_RewardCueOn.iloc[big_loop]
    working_RewardCueTime_data=working_RewardCueTime_data[~np.isnan(working_RewardCueTime_data)]
    print(working_RewardCueTime_data)
    working_RewardCue_data=dataframe_RewardCue.iloc[big_loop]



    #self_reward_data=working_pupil_data.iloc[[0,1,2,5,6,9,14,19,21,22,23,25,27]]
    #other_reward_data=working_pupil_data.iloc[[3,4,7,8,10,11,12,13,15,16,17,18,20,24,26,28,29]]
    
    
    """plotting raw pupil series after NaNs removal."""
    
    #self_reward_series=self_reward_data.iloc[i]
    
    f=plt.figure(1)
    f.set_figheight(10) #Setting the dimensions (length & width) of the figure to be generated
    f.set_figwidth(20)  
    time_axis = np.arange(0,(working_pupil_data.size))
    plt.plot(time_axis,working_pupil_data,linestyle="dotted",alpha=1)
    plt.xlabel("Time in milliseconds",fontsize="20")
    plt.ylabel("Pupil Diameter in arbitrary units",fontsize="20")
    
    
    
       
    
    """smoothening the data by using savgol;depends on whether data is noisy or not."""
    filtered_pupil_data=savgol_filter(working_pupil_data,11,3)
    print(filtered_pupil_data)
    
    #print(filtered_pupil_data)



    """plotting the smoothened data just to check."""
    
    f=plt.figure(2)
    f.set_figheight(10) 
    f.set_figwidth(20)  
    time_axis = np.arange(0,(filtered_pupil_data.size))
    plt.plot(time_axis,filtered_pupil_data,linestyle="dotted")
    plt.xlabel("Time in milliseconds",fontsize="20")
    plt.ylabel("Pupil Diameter in arbitrary units",fontsize="20")
    
    """blinks are depicted by abrupt change in pupil size,to be precise, decrease in pupil size.
    s.mathot(2013):there is no objective way to represent blinks in pupil data
    sometimes blinks are depicted by loss of signals as in NaNs,sometimes by zero and sometimes by plateauing at a constant value.
    here just to be sure,we will corroborate it with blinks detected in EyeX EyeY array of the same trial.
    After we do it, there are 2 ways to detect blinks,one is to use velocity criterion and 2nd one is to use the value at which the 
    the value at which the signal plateaus."""
    #calculating the velocity from the smoothened array by using np.diff 
    
    velocity_series=np.diff(filtered_pupil_data)
    print(velocity_series)
    
    #plotting the velocity vs time graph just to check along with pupil diameter time series.
    f=plt.figure(3)
    f.set_figheight(10)
    f.set_figwidth(20)
    time_axis=np.arange(0,(velocity_series.size))
    plt.plot(time_axis,velocity_series,linestyle="dotted")
    plt.xlabel("Time in ms",fontsize="20")
    plt.ylabel("Velocity au/sec")
    f=plt.figure(3)
    f.set_figheight(10) 
    f.set_figwidth(20)  
    time_axis = np.arange(0,(filtered_pupil_data.size))
    plt.plot(time_axis,filtered_pupil_data,linestyle="dotted")
    plt.xlabel("Time in milliseconds",fontsize="20")
    plt.ylabel("Pupil Diameter in arbitrary units",fontsize="20")
    
    """point to be noted-once we have corroborated the position of blinks from the EyeY and EyeX arrays
    and moreover we have compared both velocity-time and pupil diameter series with each other,we 
    have now triangulated the position of blinks in the pupil diameter time-series.
    One can use velocity criterion or the plateau criterion to detect blinks.
    From eyeballing, one can see that plateau criterion seems like a more favorable option.
    We shall proceed with plateau criterion."""
    
    
    bfring=100
    blink_indices=np.nonzero(filtered_pupil_data<-4.6)
    blink_indices=(np.array(blink_indices)).flatten()
    blinklist = [] # Empty list created to store blinks
    for i in blink_indices:
        b = list(range((i-bfring),(i+bfring))) # Adding fringes on either side of points exceeding blink threshold...
        blinklist += b #...and adding indices of all these points to the list of blinks
    for i in range(len(blinklist)): # Removing portions of fringes that might have dropped below zero
        if blinklist[i]<0:
            blinklist[i]=0
    final_blinks = np.unique(np.array(blinklist)) # Storing final blink indices in an array, keeping only unique values
    
    permanent_finalblinks=[]
    for i in final_blinks:
        if i < len(filtered_pupil_data):
            permanent_finalblinks.append(i)
    print(permanent_finalblinks)        
    permanent_finalblinks = np.array(permanent_finalblinks)       
        
            
        
    
    f=plt.figure(4)
    f.set_figheight(10)
    f.set_figwidth(20)
    time_axis = np.arange(0,(filtered_pupil_data.size))
    plt.plot(time_axis, filtered_pupil_data)
    plt.vlines(permanent_finalblinks,ymin = min(filtered_pupil_data), ymax = max(filtered_pupil_data), colors = 'gray')
    
      
    for i in permanent_finalblinks:
        filtered_pupil_data[i]=np.nan
        
    f=plt.figure(5)
    f.set_figheight(10)
    f.set_figwidth(20)
    time_axis = np.arange(0,(filtered_pupil_data.size))
    plt.plot(time_axis, filtered_pupil_data)
    plt.xlabel("Time in milliseconds",fontsize="20")
    plt.ylabel("Pupil Diameter in arbitrary units",fontsize="20")
    #plt.legend(loc="upper left",fontsize="16")
    plt.title("NaNs Inserted in place of blinks")
    
    ok = ~np.isnan(filtered_pupil_data)
    xp = ok.ravel().nonzero()[0]
    fp = filtered_pupil_data[~np.isnan(filtered_pupil_data)]
    x  = np.isnan(filtered_pupil_data).ravel().nonzero()[0]
    
    # Replacing nan values
    filtered_pupil_data[np.isnan(filtered_pupil_data)] = np.interp(x, xp, fp)
    
    
    f=plt.figure(6)
    f.set_figheight(10)
    f.set_figwidth(20)
    time_axis = np.arange(0,(filtered_pupil_data.size))
    plt.plot(time_axis, filtered_pupil_data )
    plt.xlabel("Time in milliseconds",fontsize="20")
    plt.ylabel("Pupil Diameter in arbitrary units",fontsize="20")
    #plt.legend(loc="upper left",fontsize="16")
    plt.title("Interpolated over NaNs")
    
    import statistics as stats
    baseline=filtered_pupil_data[0:540]
    avg=stats.mean(baseline)
    baseline_corrected= filtered_pupil_data-avg
       
    f=plt.figure(7)
    f.set_figheight(10)
    f.set_figwidth(20)
    time_axis=np.arange(0,(baseline_corrected.size))
    plt.plot(time_axis,baseline_corrected,linestyle="dotted")
    plt.xlabel("Time in milliseconds",fontsize="20")
    plt.ylabel("Pupil Diameter in arbitrary units",fontsize="20")
    plt.title("Subtractive Baseline Correction")

    baseline_avg_array.append(avg)

print(baseline_avg_array)    

baseline_avg_array=np.array(baseline_avg_array)

fig, ax = plt.subplots(figsize =(20, 10))
ax.hist(baseline_avg_array)
#plt.title("Histogram of Mean Baseline Size ")   
    
##########################################################################################################################33




filtered_self_reward_data_series=savgol_filter(self_reward_data_series, 11, 3)
filtered_other_reward_data_series=savgol_filter(other_reward_data_series, 11, 3)


f=plt.figure(8)
f.set_figheight(10) 
f.set_figwidth(20)  
time_axis = np.arange(0,(filtered_self_reward_data_series.size))
plt.plot(time_axis,filtered_self_reward_data_series,'bo',linestyle="dotted",label="Self Reward")

f=plt.figure(8)
f.set_figheight(10) 
f.set_figwidth(20)
time_axis = np.arange(0,(filtered_other_reward_data_series.size))
plt.plot(time_axis,filtered_other_reward_data_series,'ro',linestyle="dotted",label="Other Reward")
plt.xlabel("Time in milliseconds",fontsize="20")
plt.ylabel("Pupil Diameter in arbitrary units",fontsize="20")
plt.legend(loc="upper left",fontsize="16") 
plt.title("Pupil Size Time Series Comparison-Self vs Other")




bfring=100
blink_indices_O=np.nonzero(filtered_other_reward_data_series<-4.8)
blink_indices_O=(np.array(blink_indices_O)).flatten()
blinklist_O = [] # Empty list created to store blinks
for i in blink_indices_O:
    b = list(range((i-bfring),(i+bfring))) # Adding fringes on either side of points exceeding blink threshold...
    blinklist_O += b #...and adding indices of all these points to the list of blinks
for i in range(len(blinklist_O)): # Removing portions of fringes that might have dropped below zero
    if blinklist_O[i]<0:
        blinklist_O[i]=0
final_blinks_O = np.unique(np.array(blinklist_O)) # Storing final blink indices in an array, keeping only unique values

permanent_finalblinks_O=[]
for i in final_blinks_O:
    if i < len(filtered_other_reward_data_series):
        permanent_finalblinks_O.append(i)
print(permanent_finalblinks_O)        
permanent_finalblinks_O = np.array(permanent_finalblinks_O) 


bfring=100
blink_indices_S=np.nonzero(filtered_self_reward_data_series<-4.8)
blink_indices_S=(np.array(blink_indices_S)).flatten()
blinklist_S = [] # Empty list created to store blinks
for i in blink_indices_S:
    b = list(range((i-bfring),(i+bfring))) # Adding fringes on either side of points exceeding blink threshold...
    blinklist_S += b #...and adding indices of all these points to the list of blinks
for i in range(len(blinklist_S)): # Removing portions of fringes that might have dropped below zero
    if blinklist_S[i]<0:
        blinklist_S[i]=0
final_blinks_S = np.unique(np.array(blinklist_S)) # Storing final blink indices in an array, keeping only unique values

permanent_finalblinks_S=[]
for i in final_blinks_S:
    if i < len(filtered_self_reward_data_series):
        permanent_finalblinks_S.append(i)
print(permanent_finalblinks_S)        
permanent_finalblinks_S = np.array(permanent_finalblinks_S) 

    
    
    
    
    
    
f=plt.figure(9)
f.set_figheight(10)
f.set_figwidth(20)
time_axis = np.arange(0,(filtered_self_reward_data_series.size))
plt.plot(time_axis, filtered_self_reward_data_series,"bo",linestyle="dotted",label="Self Reward")
plt.vlines(permanent_finalblinks_S,ymin = min(filtered_self_reward_data_series), ymax = max(filtered_self_reward_data_series), colors = 'gray')

                               
f=plt.figure(9)
f.set_figheight(10)
f.set_figwidth(20)
time_axis = np.arange(0,(filtered_other_reward_data_series.size))
plt.plot(time_axis, filtered_other_reward_data_series,"ro",linestyle="dotted",label="Other Reward")
plt.vlines(permanent_finalblinks_O,ymin = min(filtered_other_reward_data_series), ymax = max(filtered_other_reward_data_series), colors = 'gray')
plt.xlabel("Time in milliseconds",fontsize="20")
plt.ylabel("Pupil Diameter in arbitrary units",fontsize="20")
plt.legend(loc="upper left",fontsize="16")
plt.title("NaNs Inserted over Blonks")



for i in permanent_finalblinks_S:
    filtered_self_reward_data_series[i]=np.nan
                              
for i in permanent_finalblinks_O:
    filtered_other_reward_data_series[i]=np.nan 



ok_S = ~np.isnan(filtered_self_reward_data_series)
xp_S = ok_S.ravel().nonzero()[0]
fp_S = filtered_self_reward_data_series[~np.isnan(filtered_self_reward_data_series)]
x_S  = np.isnan(filtered_self_reward_data_series).ravel().nonzero()[0] 

filtered_self_reward_data_series[np.isnan(filtered_self_reward_data_series)] = np.interp(x_S, xp_S, fp_S)
  
        
    
ok_O = ~np.isnan(filtered_other_reward_data_series)
xp_O = ok_O.ravel().nonzero()[0]
fp_O = filtered_other_reward_data_series[~np.isnan(filtered_other_reward_data_series)]
x_O  = np.isnan(filtered_other_reward_data_series).ravel().nonzero()[0]   
  
  
  
filtered_other_reward_data_series[np.isnan(filtered_other_reward_data_series)] = np.interp(x_O, xp_O, fp_O)
  
    
    
    
f=plt.figure(10)
f.set_figheight(10)
f.set_figwidth(20)
time_axis = np.arange(0,(filtered_self_reward_data_series.size))
plt.plot(time_axis, filtered_self_reward_data_series,"bo",linestyle="dotted",label="Self Reward")    

f=plt.figure(10)
f.set_figheight(10)
f.set_figwidth(20)
time_axis = np.arange(0,(filtered_other_reward_data_series.size))
plt.plot(time_axis, filtered_other_reward_data_series,"ro",linestyle="dotted",label="Other Reward")    
plt.xlabel("Time in milliseconds",fontsize="20")
plt.ylabel("Pupil Diameter in arbitrary units",fontsize="20")
plt.legend(loc="upper left",fontsize="16")
plt.title("Interpolated over NaNs")


baseline_S=filtered_self_reward_data_series[0:50]
avg_s=stats.mean(baseline_S)
baseline_corrected_S= filtered_self_reward_data_series-avg_s
   
f=plt.figure(11)
f.set_figheight(10)
f.set_figwidth(20)
time_axis=np.arange(0,(baseline_corrected_S.size))
plt.plot(time_axis,baseline_corrected_S,"bo",linestyle="dotted",label="Self Reward")


baseline_O=filtered_other_reward_data_series[0:50]
avg_o=stats.mean(baseline_O)
baseline_corrected_O=filtered_other_reward_data_series -avg_o
   
f=plt.figure(11)
f.set_figheight(10)
f.set_figwidth(20)
time_axis=np.arange(0,(baseline_corrected_O.size))
plt.plot(time_axis,baseline_corrected_O,"ro",linestyle="dotted",label="Other Reward")
plt.xlabel("Time in milliseconds",fontsize="20")
plt.ylabel("Change in Pupil size over time wrt Baseline size",fontsize="20")
plt.legend(loc="upper left",fontsize="16")
plt.title("Subtractive Baseline Corrected Pupil Size Series")



#proportion change

proportion_change_self=(baseline_corrected_S/avg_s)*100
proportion_change_other=(baseline_corrected_O/avg_o)*100



f=plt.figure(12)
f.set_figheight(10)
f.set_figwidth(20)
time_axis=np.arange(0,(baseline_corrected_O.size))
plt.plot(time_axis,proportion_change_other,'ro',linestyle="dotted",label="Other Reward")

f=plt.figure(12)
f.set_figheight(10)
f.set_figwidth(20)
time_axis=np.arange(0,(baseline_corrected_S.size))
plt.plot(time_axis,proportion_change_self,'bo',linestyle="dotted",label="Self Reward")
plt.xlabel("Time in milliseconds",fontsize="20")
plt.ylabel("Percentage Change in Pupil Size from the Baseline Size")
plt.legend(loc="upper left",fontsize="16")
plt.title("Percentage Change in Pupil Size")



self_reward_data_series_cut=filtered_self_reward_data_series[1080:]

other_reward_data_series_cut=filtered_other_reward_data_series[1130:]

baseline_S_cut=self_reward_data_series_cut[0:50]
baseline_O_cut=other_reward_data_series_cut[0:50]


avg_s_cut=stats.mean(baseline_S_cut)
baseline_corrected_S_cut=self_reward_data_series_cut-avg_s_cut

avg_o_cut=stats.mean(baseline_O_cut)
baseline_corrected_O_cut= other_reward_data_series_cut-avg_o_cut

proportion_change_self_cut=(baseline_corrected_S_cut/avg_s_cut)*100
proportion_change_other_cut=(baseline_corrected_O_cut/avg_o_cut)*100



f=plt.figure(13)
f.set_figheight(10)
f.set_figwidth(20)
time_axis=np.arange(0,(baseline_corrected_O_cut.size))
plt.plot(time_axis,proportion_change_other_cut,'ro',linestyle="dotted",label="Other Reward")

f=plt.figure(13)
f.set_figheight(10)
f.set_figwidth(20)
time_axis=np.arange(0,(baseline_corrected_S_cut.size))
plt.plot(time_axis,proportion_change_self_cut,'bo',linestyle="dotted",label="Self Reward")
plt.xlabel("Time in milliseconds after Reward Cue",fontsize="20")
plt.ylabel("Percentage Change in Pupil Size from the Baseline Size")
plt.legend(loc="upper left",fontsize="16")
plt.title("Percentage Change in Pupil Size after Reward Cue")


final_reward=list(filtered_self_reward_data_series)
final_reward.extend(list(filtered_other_reward_data_series))


mean_final_reward=stats.mean(final_reward)

std_final=stats.stdev(final_reward)

zscored=[]
for i in range(len(final_reward)):
    z=(final_reward[i]-mean_final_reward)/std_final
    zscored.append(z)
    
self_Z=zscored[:3878]
other_z=zscored[3877:]



self_Z=np.array(self_Z)
other_z=np.array(other_z)

baseline_zs=self_Z[0:50]
baseline_zo=other_z[0:50]


baseline_zs=np.array(baseline_zs)
baseline_zo=np.array(baseline_zo)

mean_base_s=stats.mean(baseline_zs)
mean_base_o=stats.mean(baseline_zo)


final_z_b_s=self_Z-mean_base_s
final_z_b_o=other_z-mean_base_o
percentage_S=(final_z_b_s/mean_base_s)*100

percentage_o=(final_z_b_o/mean_base_o)*100

f=plt.figure(14)
f.set_figheight(10)
f.set_figwidth(20)
time_axis=np.arange(0,(other_z.size))
plt.plot(time_axis,percentage_o,'ro',linestyle="dotted",label="Other Reward")

f=plt.figure(14)
f.set_figheight(10)
f.set_figwidth(20)
time_axis=np.arange(0,(self_Z.size))
plt.plot(time_axis,percentage_S,'bo',linestyle="dotted",label="Self Reward")
plt.xlabel("Time in milliseconds",fontsize="20")
plt.ylabel("Percentage change in pupil size")
plt.legend(loc="upper left",fontsize="16")
plt.title("Percentage change in pupil size")



#######################################################################################################















