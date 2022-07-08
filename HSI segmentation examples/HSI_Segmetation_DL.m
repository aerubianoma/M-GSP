% Example of HSI image segmentation

% Load the indian_pines data set consisting of a single hyperspectral
% image of size 145X145 with 220 color channels
hcube = hypercube("indian_pines.dat");
rgbImg = colorize(hcube,method="rgb");
imshow(rgbImg)

% Load the ground truth labels and specify the number of classes
gtLabel = load("indian_pines_gt.mat");
gtLabel = gtLabel.indian_pines_gt;
numClasses = 16;

% Preparing the train data, this reduces the number of spectral brands to
% the 30 most representative
dimReduction = 30;
imageData = hyperpca(hcube,dimReduction);

% Normalize the data
sd = std(imageData,[],3);
imageData = imageData./sd;

% Split the hyperspectral image into patches of size 25-by-25 pixels with
% 30 channels, also returns a single label for each patch, which is the label of the central pixel.
windowSize = 25;
inputSize = [windowSize windowSize dimReduction];
[allPatches,allLabels] = createImagePatchesFromHypercube(imageData,gtLabel,windowSize);
indianPineDataTransposed = permute(allPatches,[2 3 4 1]);
dsAllPatches = augmentedImageDatastore(inputSize,indianPineDataTransposed,allLabels);


% Not all of the cubes in this data set have labels. However, training the network 
% requires labeled data. Select only the labeled cubes for training. Count how many labeled 
% patches are available.
patchesLabeled = allPatches(allLabels>0,:,:,:);
patchLabels = allLabels(allLabels>0);
numCubes = size(patchesLabeled,1);

% Convert the numerical labels to categorical
patchLabels = categorical(patchLabels);

% Divide the patches into training and test data sets
[trainingIdx,valIdx,testIdx] = dividerand(numCubes,0.3,0,0.7);
dataInputTrain = patchesLabeled(trainingIdx,:,:,:);
dataLabelTrain = patchLabels(trainingIdx,1);
dataInputTest = patchesLabeled(testIdx,:,:,:);
dataLabelTest = patchLabels(testIdx,1);

% Transpose the input data.
dataInputTransposeTrain = permute(dataInputTrain,[2 3 4 1]); 
dataInputTransposeTest = permute(dataInputTest,[2 3 4 1]);

% Create datastores that read batches of training and test data.
dsTrain = augmentedImageDatastore(inputSize,dataInputTransposeTrain,dataLabelTrain);
dsTest = augmentedImageDatastore(inputSize,dataInputTransposeTest,dataLabelTest);

% Create CSCNN Classification Network
layers = [
    image3dInputLayer(inputSize,Name="Input",Normalization="None")
    convolution3dLayer([3 3 7],8,Name="conv3d_1")
    reluLayer(Name="Relu_1")
    convolution3dLayer([3 3 5],16,Name="conv3d_2")
    reluLayer(Name="Relu_2")
    convolution3dLayer([3 3 3],32,Name="conv3d_3")
    reluLayer(Name="Relu_3")
    convolution3dLayer([3 3 1],8,Name="conv3d_4")
    reluLayer(Name="Relu_4")
    fullyConnectedLayer(256,Name="fc1")
    reluLayer(Name="Relu_5")
    dropoutLayer(0.4,Name="drop_1")
    fullyConnectedLayer(128,Name="fc2")
    dropoutLayer(0.4,Name="drop_2")
    fullyConnectedLayer(numClasses,Name="fc3")
    softmaxLayer(Name="softmax")
    classificationLayer(Name="output")];
lgraph = layerGraph(layers);

% Visualizing the network using Deep Network Designer
deepNetworkDesigner(lgraph)

% Specify the required network parameters. For this example, train the network for 
% 100 epochs with an initial learning rate of 0.001, a batch size of 256, and 
% Adam optimization.

numEpochs = 100;
miniBatchSize = 256;
initLearningRate = 0.001;
momentum = 0.9;
learningRateFactor = 0.01;

options = trainingOptions("adam", ...
    InitialLearnRate=initLearningRate, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=30, ...
    LearnRateDropFactor=learningRateFactor, ...
    MaxEpochs=numEpochs, ...
    MiniBatchSize=miniBatchSize, ...
    GradientThresholdMethod="l2norm", ...
    GradientThreshold=0.01, ...
    VerboseFrequency=100, ...
    ValidationData=dsTest, ...
    ValidationFrequency=100);

% Train the network
doTraining = true;
if doTraining
    net = trainNetwork(dsTrain,lgraph,options);
    modelDateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
    save("trainedIndianPinesCSCNN-"+modelDateTime+".mat","net");
else
    dataDir = pwd;
    trainedNetwork_url = "https://ssd.mathworks.com/supportfiles/image/data/trainedIndianPinesCSCNN.mat";
    downloadTrainedNetwork(trainedNetwork_url,pwd);
    load(fullfile(dataDir,"trainedIndianPinesCSCNN.mat"));
end

% Classify Hyperspectral Image Using Trained CSCNN
% Calculate the accuracy of the classification for the test data set. Here, accuracy 
% is the fraction of the correct pixel classification over all the classes.
predictionTest = classify(net,dsTest);
accuracy = sum(predictionTest == dataLabelTest)/numel(dataLabelTest);
disp("Accuracy of the test data = "+num2str(accuracy))

% Reconstruct the complete image by classifying all image pixels, including pixels in 
% labeled training patches, pixels in labeled test patches, and unlabeled pixels
prediction = classify(net,dsAllPatches);
prediction = double(prediction);

% The network is trained on labeled patches only. Therefore, the predicted 
% classification of unlabeled pixels is meaningless. Find the unlabeled patches 
% and set the label to 0.
patchesUnlabeled = find(allLabels==0);
prediction(patchesUnlabeled) = 0;

% Reshape the classified pixels to match the dimensions of the ground truth image.
[m,n,d] = size(imageData);
indianPinesPrediction = reshape(prediction,[n m]);
indianPinesPrediction = indianPinesPrediction';

% Display the ground truth and predicted classification.
cmap = parula(numClasses);

figure
tiledlayout(1,2,TileSpacing="Tight")
nexttile
imshow(gtLabel,cmap)
title("Ground Truth Classification")

nexttile
imshow(indianPinesPrediction,cmap)
colorbar
title("Predicted Classification")

%function that creates patches in the image
function [patchData,patchLabel] = createImagePatchesFromHypercube(hcube,groundTruthLabel,winSize)

padding = floor((winSize-1)/2);
zeroPaddingPatch = padarray(hcube,[padding,padding],0,'both');

[rows,cols,ch] = size(hcube);
patchData = zeros(rows*cols,winSize,winSize,ch);
patchLabel = zeros(rows*cols,1);
zeroPaddedInput = size(zeroPaddingPatch);
patchIdx = 1;
for i= (padding+1):(zeroPaddedInput(1)-padding)
    for j= (padding+1):(zeroPaddedInput(2)-padding)
        patch = zeroPaddingPatch(i-padding:i+padding,j-padding:j+padding,:);
        patchData(patchIdx,:,:,:) = patch;
        patchLabel(patchIdx,1) = groundTruthLabel(i-padding,j-padding);
        patchIdx = patchIdx+1;
    end
end

end