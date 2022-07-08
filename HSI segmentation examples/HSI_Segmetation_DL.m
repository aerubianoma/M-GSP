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