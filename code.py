##################TRAINING############################
import tensorflow as tf
from scipy.special import softmax
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

# Part 1 - Data Preprocessing

# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('/content/drive/MyDrive/CAPSTONE/dataset/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('/content/drive/MyDrive/CAPSTONE/dataset/test_set',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', input_shape=[128, 128, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a third convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a fourth convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(3, activation='softmax'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 30)

cnn.save('/content/drive/MyDrive/CAPSTONE/Model')




#####################RESULTS#########################


#importing of various libraries required for code
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
cm = sns.diverging_palette(-5, 5, as_cmap=True)



#importing the csv file of the animal population
df = pd.read_csv(r'/content/drive/MyDrive/CAPSTONE/population.csv');

results={
   0:'Buffalo',
   1:'Elephant',
   2:'Rhinoceros'
}
from PIL import Image
import numpy as np
im=Image.open("/content/drive/MyDrive/CAPSTONE/dataset/single_prediction/r2.jpg")
# the input image is required to be in the shape of dataset, i.e (32,32,3)

model = tf.keras.models.load_model("/content/drive/MyDrive/CAPSTONE/Model")

im=im.resize((128,128))
im=np.expand_dims(im,axis=0)
im=np.array(im)
pred=model.predict_classes([im])[0]
print(pred,results[pred])

a=pred


#Showing the population and year of census taken for the animal chosen (Detected from CNN model)

x=df[['Population','Population.1','Population.2','Population.3']]
x1=x.loc[a]
y=df[['Year','Year.1','Year.2','Year.3']]
y1=y.loc[a]

"""
print(x1);
print(" ");
print(y1);

"""

#Showing the details of the specific animal with the animal name as an additional column
#Animal name is only thing added when compared to the code above

x2=df.loc[[a],['Animal','Population','Population.1','Population.2','Population.3']]
y2=df.loc[[a],['Animal','Year','Year.1','Year.2','Year.3']]

"""
print(x2);
print(" ");
print(y2);

"""

#Dropping the Animal column from both the extracted rows, so that only required information for plotting graph is saved

x3=x2.drop('Animal',axis=1)
x4=x3.transpose()
y3=y2.drop('Animal',axis=1)
y4=y3.transpose()

"""
print(x4);
print(y4);

"""

#Displaying output of the animal population using bar graph


if (a==0):
    w = 'Asiatic Buffaloes';
elif (a==1):
    w = 'Elephants';
elif (a==2):
    w = 'Rhinoceros';
elif (a==3):
    w = 'Bengal Tigers';
elif (a==4):
    w = 'Swamp Deer';
plt.bar(y4[a],x4[a],color = "blue")
plt.xlabel("Year", fontsize=20)
plt.ylabel("Population",fontsize=20)
plt.xticks(rotation = 90)
plt.title("Animal: "+ w, fontsize=25)
plt.show();
z1=df.loc[a,'Initial Condition']
print("Initial condition: ", z1);
print(" ");
print(" ");



#==================================================================



#calculating the change percentage in population of detected animal through mentioned years

x3a=df.loc[a,"Population"]
x3b=df.loc[a,"Population.1"]
x3c=df.loc[a,"Population.2"]
x3d=df.loc[a,"Population.3"]

y3a=df.loc[a,'Year']
y3b=df.loc[a,'Year.1']
y3c=df.loc[a,'Year.2']
y3d=df.loc[a,'Year.3']


#Storing the population increment/decrement percentage in variables
#x4d is for storing total population change, ie population difference between first and last available population census

inri_decri = [0,0,0,0]
x4a =((x3b/x3a)-1)*100
x4b =((x3c/x3b)-1)*100
x4c =((x3d/x3c)-1)*100
x4d =((x3d/x3a)-1)*100

#change is stored in a list form to make it easier to add in the table
Change_percentage = [x4a,x4b,x4c,x4d]

#making lists for storing the prediction chances for the animals
negative=[]
positive=[]
neutral=[]
for j in range(len(Change_percentage)):
    if (Change_percentage[j]<0):
        negative.append(1)
    elif(Change_percentage[j]==0):
        neutral.append(1)
    else:
        positive.append(1)

# (a)steps to be taken if negative list comes into picture        
if(len(negative)>=2):
    pred = 'Has very less chances of survival, ie an endangered species'
elif(len(negative)>=1):
    pred = 'moderate to good chances of survival'

#(b)steps taken if positive but increment percentage less than 3 in any subsequent census
if(len(positive)==4):
    for k in range(len(Change_percentage)):
        if(Change_percentage[k]<=3):
            pred = 'Moderate to high chances of survival'
            continue
        else:
            pred = 'Very high chances of survival (growth rate in the past was always high) '
            continue
       

#for loop is run for increment decrement column in the table, to decide whether the population increased or decreased
#during the mentioned time interval in years

for i in range((len(Change_percentage))):
   
    if (Change_percentage[i]<0):
        inri_decri[i] = "Decrement"
    elif(Change_percentage[i]==0):
        inri_decri[i] = "No change"
    else:
        inri_decri[i] = "Increment"
       
#Storing starting & ending years in 2 different arrays, which will be used to fill two columns of the table

Year_start = np.array([y3a,y3b,y3c,y3a])
Year_end   = np.array([y3b,y3c,y3d,y3d])
Start_end  = list(zip(Year_start,Year_end,Change_percentage,inri_decri))

Pop_stats = pd.DataFrame(Start_end, columns = ['Year (from)','Year (to)','Population Change (%)','Population +/-'])
Pop_stats.style.background_gradient(cmap=cm, subset=Pop_stats.index[3])

print(Pop_stats);
print(" ");
print(" ");
#def highlight_cols(x):
     
    # copy df to new - original data is not changed
    #Pop_stats = x.copy()
     
    ## select all values to green color
    #Pop_stats.loc[:,:] = 'background-color: green'
 
    # return color df
    #return Pop_stats

#display(Pop_stats.style.apply(highlight_cols, axis = None))

#Pop_stats



#================================================================

#Future Trends of the detected animal done assuming constant rate of + or -

year_change=y3d-y3a
Total_pop_change=Change_percentage[3]
avg_change_annually=Total_pop_change/year_change

intial_end_pop = x3d
est_pop = [0,0,0,0]

#for loop is run for calculating the future population

for i in range((len(Change_percentage))):
    x = intial_end_pop*(1+(Total_pop_change)/100)
    est_pop[i]=x
    intial_end_pop=x
est_pop_rounded = [round(num) for num in est_pop]

#Year_start_predict = np.array([y3d,y3d+year_change,y3d+2*(year_change),y3d+3*(year_change)])
Year_end_predict = np.array([y3d+year_change,y3d+2*(year_change),y3d+3*(year_change),y3d+4*(year_change)])
Start_end_predict  = list(zip(Year_end_predict,est_pop_rounded))

Pop_stats_pred = pd.DataFrame(Start_end_predict, columns = ['Year','Population (Estimated)'])

print(Pop_stats_pred);
print(" ");
print(" ");


#=================================================================


#Graphical Representation of future estimated population of detected animal

plt.rc('lines', linewidth=1, linestyle='-', marker='*')
plt.rcParams["figure.figsize"] = (6, 6)
plt.plot(Year_end_predict, est_pop_rounded, 'r')
plt.xlabel('Year',fontsize=15)
plt.ylabel('Population',fontsize=15)
plt.title('Future population trends of detected animal (Assuming constant rate)',fontsize=20)
plt.show()
print("Possibility of surviving in the future: ",pred) 
print("")


df_pre = pd.read_csv(r'/content/drive/MyDrive/CAPSTONE/Precautions.csv');
prec=df_pre.loc[[a],['Animal','Precaution 1','Precaution 2','Precaution 3']]
prec_1=df_pre.loc[a,"Precaution 1"]
prec_2=df_pre.loc[a,"Precaution 2"]
prec_3=df_pre.loc[a,"Precaution 3"]
prec_list = [prec_1,prec_2,prec_3]
print("The precautions to be taken to prevent extinction of", w ,"are : " )
for i in range(len(prec_list)):
    print(i+1, ". ", prec_list[i])