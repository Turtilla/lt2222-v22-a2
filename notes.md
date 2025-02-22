### Part 1
For implementing randomness in the sample_lines() function I made use of the [random](https://docs.python.org/3/library/random.html) module from the Python Standard Library. Using the randint() function from that module, I generate a number Y from 0 to X, where X is the total number of lines minus the desired number of lines. Then I use that generated number Y to be the start of the slice, and that number Y plus the desired number of lines as the end. _Note that the constraint on generating the number Y is due to it otherwise potentially leading to an IndexError if Y + desired number of lines exceeded the length of the file, so to say._

### Part 3
I just wanted to remark here that I shuffled the samples before putting them in the dataframe (but after having based the columns on them). I wanted to do this to avoid any bias resulting from the fact that they come in the order in which they were found in the text. Since I only shuffle the order of the vectors and not their contents, they still represent the same samples, it is just the order of the samples that is changed.

### Part 6
In this part I got the following results while running the script through the notebook on the mltgpu server - but of course there is randomness to it so the results cannot be repeated. These results can be found in the python notebook in the repo.   

>For the model SVC(kernel='linear') the following scores were obtained:      
>	precision = 0.532258064516129    
>	recall = 0.039663461538461536    
>	f1 score = 0.07382550335570469   
>For the model SVC() the following scores were obtained:    
>	precision = 0.8441558441558441    
>	recall = 0.078125     
>	f1 score = 0.14301430143014301      

The latter one is the rbf model, but since rbf is the default setting of SVC(), it does not show up in the description. It is also worth noting that training the models on such big training sets took about 2h, so do not be alarmed if it does not instantly work.   

Straight out of the gate it is visible that the radial basis function model performs better. It has higher scores in all three categories, with a surprisingly high precision (meaning the amount of true positives in all positives), so if something is predicted to be a verb by this model, it has an 84% chance of actually being a verb. Both models, however, have a very low recall (the amount of true positives in relation to true positives and false negatives, so all the original verbs), meaning only a small fraction of actual verbs do get identified. F1 is a mix of both of these scores, and with it being quite low as well we can probably safely assume that these two models are not the best verb-predicting machines.   

As explained in on [sklearn's website](https://scikit-learn.org/stable/modules/svm.html), and from the lectures, I understand that having a linear kernel in the support vector machine means that the divisions go more or less by straight lines, while the RBF kernel allows it to "draw curves" through the data (if we think of it visually). I think this may be the reason why the RBF (default) kernel machine did two times better than the linear one. This also implies that in this sample the data is not distributed in a linear fashion. This could also be why the model fails to detect many actual verbs, because there is little pattern to be found in the training data (but whatever there is, works better with a non-linear model).

### Bonus Part
For this part I chose to use the [nearest neighbors classification](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification) class from sklearn. I implemented it in mycode.py analogously to the train() function, but under the name of train_neighbors(). You can test how it works by running the script called high_pass.py from the command line with one command line argument: the name of or path to the file. For instance, when running this on mltgpu, I entered the following in the command line to get it to work: `python3 high_pass.py /scratch/UN-english.txt.gz`. This should print you more or less all the results like in the notebook, with the exception of using the NearestCentroid() class from sklearn instead of the SVC() models. The results I got from running it on mltgpu were the following:       

>The model's evaluation:   
>For the model NearestCentroid() the following scores were obtained:   
>        precision = 0.25585798816568045   
>        recall = 0.6676961087090797   
>        f1 score = 0.3699520876112251      

It is also quite interesting that it seems that this way of classifying the samples was much, much faster.   

One can quite clearly see that overall this classification worked "better". The F1 score is much higher than in either of the SVM models. However, the precision is much lower here, meaning that there were many false positives in the output; the model did much better at detecting the true positives out of the test data though - so it is more likely to detect verbs, but it also overdetects them, giving many false positives. Overall though, it seems that it is better at doing its job than the SVM models.   

According to what is written on sklearn's website (linked above), the nearest neighbors classification does not "learn" or "construct" some sort of an internal model (e.g. some lines or planes to divide the space in which the vectors representing the samples are located), but instead simply stores the training data and assigns the class based on a "majority vote" of what of the training data is the most similar to the test data. This means that, more or less, if in the training data a given sample, say ('hn_1','ly_2','ow_4','ox_5'), was in 75% cases a verb one, then it would classify every such example that was fed to it as a verb. This could be why it detected overall more verbs (and found more of the original ones), but also overdetected them, resulting in many false positives and a low precision score. Overall though it seems that this "majority vote" works out better than the other model, and it could be because while language can form infinite numbers of sentences and meanings, many of them repeat quite often, and so a majority vote can yield surprisingly good results. 
