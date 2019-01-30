% te importance of preprocessing
clearvars
%% load small scale - 256x256
dataSetDir = fullfile('C:\sea urchin\Hiearhical network\data\orig');
imageDir = fullfile(dataSetDir,'sparse50_nophase');
labelDir = fullfile(dataSetDir,'seg');
%% load images and labels
imds1 = imageDatastore(imageDir);
classNames = ["structure","background"];
labelIDs   = [0 255];
pxds1 = pixelLabelDatastore(labelDir,classNames,labelIDs);
%%
figure
reset(imds1)
reset(pxds1)
I = read(imds1);
C = read(pxds1);
clf
I = imresize(I,5);
L = imresize(uint8(C),5);
imshowpair(I,L,'montage')

%% Resize  Data into 128x128x3(no need to resize pixel)%_concatenate
imageFolder = fullfile(dataSetDir,'imagesResized_50_nophase_concatenate',filesep);
imds1 = resizeImages_concatenate(imds1,imageFolder);
%% preparing training and testing sets(great! no need to prepare by myself)
[imdsTrain1,imdsTest1,pxdsTrain1,pxdsTest1,trainingIdx,testIdx] = partitionData(imds1,pxds1,0.7);
numel(imdsTrain1.Files)
%%
numFilters = 64;
filterSize = 3;
numClasses = 2;
layers = [
    imageInputLayer([256 256 3])
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1);
    reluLayer()
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    convolution2dLayer(1,numClasses);
    softmaxLayer()
    pixelClassificationLayer()
    ];
% layers = createUnet([256 256 3],1);
%% training options(look into these parameters)
options = trainingOptions('adam', ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.0005, ...
    'MaxEpochs',5, ...  
    'MiniBatchSize',4, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', '', ...
    'VerboseFrequency',2,...
    'Plots','training-progress');
%     'Momentum',0.9, ...
%% data agumentation
augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);
% %% start training
pximds = pixelLabelImageDatastore(imdsTrain1,pxdsTrain1)%, ...
 %   'DataAugmentation',augmenter);
%
doTraining = 1;
if doTraining    
    [net, info] = trainNetwork(pximds,layers,options);
else
    data = load(pretrainedSegNet);
    net = data.net;
end
%%
orignet50 = net;
save('orignet50.mat','orignet50')
%% test
load orignet50
clf
reset(imdsTest1)
reset(pxdsTest1)
I = read(imdsTest1);
C = semanticseg(I, orignet50);
B = rgb2gray(labeloverlay(I,C));

testlabel = read(pxdsTest1);
Bb = rgb2gray(labeloverlay(I,testlabel));
L = imresize(uint8(testlabel),5);
%
L1 = uint8(C);
subplot(131);imagesc((I));axis image

subplot(132);imagesc(L);axis image
subplot(133);imagesc(L1);axis image;
colormap gray
%%
newimageDir = fullfile('C:\sea urchin\Hiearhical network\data\orig\combine1_50');
pxdsTestResults1 = semanticseg(imdsTest1, orignet50, "WriteLocation", newimageDir,'MiniBatchSize',16);
%
newimageDir = fullfile('C:\sea urchin\Hiearhical network\data\orig\combine1test_50');
pxdsTrainResults1= semanticseg(imdsTrain1, orignet50, "WriteLocation", newimageDir,'MiniBatchSize',4);
metrics1 = evaluateSemanticSegmentation(pxdsTestResults1, pxdsTest1);
%% downsample 2 - 128x125
dataSetDir = fullfile('C:\sea urchin\Hiearhical network\data\downsample2');
imageDir = fullfile(dataSetDir,'sparse50_nophase');
labelDir = fullfile(dataSetDir,'seg');
%% load images and labels
imds2 = imageDatastore(imageDir);
classNames = ["structure","background"];
labelIDs   = [0 255];
pxds2 = pixelLabelDatastore(labelDir,classNames,labelIDs);
%%
clf
I = read(imds2);
C = read(pxds2);
I = imresize(I,5);
L = imresize(uint8(C),5);
imshowpair(I,L,'montage')
%% Resize  Data into 128x128x3(no need to resize pixel)
imageFolder = fullfile(dataSetDir,'imagesResized_50_nophase_concatenate',filesep);
imds2 = resizeImages_concatenate(imds2,imageFolder);
%% preparing training and testing sets(great! no need to prepare by myself)
trainingImages = imds2.Files(trainingIdx);
testImages = imds2.Files(testIdx);
imdsTrain2 = imageDatastore(trainingImages);
imdsTest2 = imageDatastore(testImages);
% Extract class and label IDs info.
classes = pxds2.ClassNames;
labelIDs = [0 255];

% Create pixel label datastores for training and test.
trainingLabels = pxds2.Files(trainingIdx);
testLabels = pxds2.Files(testIdx);
pxdsTrain2 = pixelLabelDatastore(trainingLabels, classes, labelIDs);
pxdsTest2 = pixelLabelDatastore(testLabels, classes, labelIDs);
numel(imdsTrain2.Files)
%%
numFilters = 64;
filterSize = 3;
numClasses = 2;
layers = [
    imageInputLayer([128 128 3])
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1);
    reluLayer()
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    convolution2dLayer(1,numClasses);
    softmaxLayer()
    pixelClassificationLayer()
    ];
%% training options(look into these parameters)
options = trainingOptions('adam', ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.0005, ...
    'MaxEpochs',5, ...  
    'MiniBatchSize',16, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', '', ...
    'VerboseFrequency',2,...
    'Plots','training-progress');
%     'Momentum',0.9, ...
%% data agumentation
augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);
%% start training
pximds = pixelLabelImageDatastore(imdsTrain2,pxdsTrain2)%, ...
 %   'DataAugmentation',augmenter);
%
doTraining = 1;
if doTraining    
    [net, info] = trainNetwork(pximds,layers,options);
else
    data = load(pretrainedSegNet);
    net = data.net;
end
%%
downsample2net50 = net;
save('downsample2net50.mat','downsample2net50')
%% test
load downsample2net50
clf
% reset(imdsTest2)
% reset(pxdsTest2)
I = read(imdsTest2);
C = semanticseg(I, downsample2net50);
B = rgb2gray(labeloverlay(I,C));

testlabel = read(pxdsTest2);
Bb = rgb2gray(labeloverlay(I,testlabel));
L = imresize(uint8(testlabel),5);
%
L1 = uint8(C);
subplot(131);imagesc((I));axis image

subplot(132);imagesc(L);axis image
subplot(133);imagesc(L1);axis image;
colormap gray
%%
newimageDir = fullfile('C:\sea urchin\Hiearhical network\data\downsample2\combine2_50');
pxdsTestResults2 = semanticseg(imdsTest2, downsample2net50, "WriteLocation", newimageDir,'MiniBatchSize',8);
newimageDir = fullfile('C:\sea urchin\Hiearhical network\data\downsample2\combine2test_50');
pxdsTrainResults2= semanticseg(imdsTrain2, downsample2net50, "WriteLocation", newimageDir,'MiniBatchSize',8);

metrics2 = evaluateSemanticSegmentation(pxdsTestResults2, pxdsTest2);
%% downsample 4
dataSetDir = fullfile('C:\sea urchin\Hiearhical network\data\downsample4');
imageDir = fullfile(dataSetDir,'sparse50_nophase');
labelDir = fullfile(dataSetDir,'seg');
%% load images and labels
imds4 = imageDatastore(imageDir);
classNames = ["structure","background"];
labelIDs   = [0 255];
pxds4 = pixelLabelDatastore(labelDir,classNames,labelIDs);
%% Resize  Data into 128x128x3(no need to resize pixel)
imageFolder = fullfile(dataSetDir,'imagesResized_50_nophase_concatenate',filesep);
imds4 = resizeImages_concatenate(imds4,imageFolder);
%%
clf
I = read(imds4);
C = read(pxds4);
I = imresize(I,1);
L = imresize(uint8(C),1);
imshowpair(I,L,'montage')
%% preparing training and testing sets(great! no need to prepare by myself)
trainingImages = imds4.Files(trainingIdx);
testImages = imds4.Files(testIdx);
imdsTrain4 = imageDatastore(trainingImages);
imdsTest4 = imageDatastore(testImages);
% Extract class and label IDs info.
classes = pxds4.ClassNames;
labelIDs = [0 255];

% Create pixel label datastores for training and test.
trainingLabels = pxds4.Files(trainingIdx);
testLabels = pxds4.Files(testIdx);
pxdsTrain4 = pixelLabelDatastore(trainingLabels, classes, labelIDs);
pxdsTest4 = pixelLabelDatastore(testLabels, classes, labelIDs);
numel(imdsTrain4.Files)

%%
numFilters = 64;
filterSize = 3;
numClasses = 2;
layers = [
    imageInputLayer([64 64 3])
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1);
    reluLayer()
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    convolution2dLayer(1,numClasses);
    softmaxLayer()
    pixelClassificationLayer()
    ];
%% training options(look into these parameters)
options = trainingOptions('adam', ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.0005, ...
    'MaxEpochs',10, ...  
    'MiniBatchSize',32, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', '', ...
    'VerboseFrequency',2,...
    'Plots','training-progress');
%     'Momentum',0.9, ...
%% data agumentation
augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);
%% start training
pximds = pixelLabelImageDatastore(imdsTrain4,pxdsTrain4)%, ...
 %   'DataAugmentation',augmenter);
%
doTraining = 1;
if doTraining    
    [net, info] = trainNetwork(pximds,layers,options);
else
    data = load(pretrainedSegNet);
    net = data.net;
end
%%
downsample4net50 = net;
save('downsample4net50.mat','downsample4net50')
%% test
load downsample4net50
clf
%
% reset(imdsTest4)
% reset(pxdsTest4)
I = read(imdsTest4);
C = semanticseg(I, downsample4net50);
B = rgb2gray(labeloverlay(I,C));

testlabel = read(pxdsTest4);
Bb = rgb2gray(labeloverlay(I,testlabel));
L = imresize(uint8(testlabel),5);
%
L1 = uint8(C);
subplot(131);imagesc((I));axis image

subplot(132);imagesc(L);axis image
subplot(133);imagesc(L1);axis image;
colormap gray
%%
newimageDir = fullfile('C:\sea urchin\Hiearhical network\data\downsample4\combine4_50');
pxdsTestResults4 = semanticseg(imdsTest4, downsample4net50, "WriteLocation", newimageDir);
newimageDir = fullfile('C:\sea urchin\Hiearhical network\data\downsample4\combine4test_50');
pxdsTrainResults4 = semanticseg(imdsTrain4, downsample4net50, "WriteLocation", newimageDir);

%
metrics4 = evaluateSemanticSegmentation(pxdsTestResults4, pxdsTest4);
%% combine
%
reset(imdsTest1)
reset(pxdsTestResults4)
reset(pxdsTestResults2)
reset(pxdsTestResults1)
imageFolder = fullfile('C:\sea urchin\Hiearhical network\data\combine\combineTest_50');
% index=1;
while hasdata(pxdsTestResults2)
%     index = index+1;
    % Read an image.
    [I0,info] = read(imdsTest1);
    [I1,info] = read(pxdsTestResults1);
    [I2,info] = read(pxdsTestResults2);
    [I4,info] = read(pxdsTestResults4);
    %upsample
    I0 = double(I0(:,:,1));
    I0 = (I0-min(I0(:)))./(max(I0(:))-min(I0(:)));
    I1 = uint8(I1);
    I1 = (I1-min(I1(:)))./(max(I1(:))-min(I1(:)));
    I2 = imresize(uint8(I2),2);
    I2 = uint8(I2);
    I2 = (I2-min(I2(:)))./(max(I2(:))-min(I2(:)));
%     I4 = impyramid(double(I4),'expand');
    I4 = imresize(uint8(I4),4);
    I4 = uint8(I4);
    I4 = (I4-min(I4(:)))./(max(I4(:))-min(I4(:)));
    % combine
    I = zeros(256,256,4);
    I(:,:,1)=I1;
    I(:,:,2)=I2;
    I(:,:,3)=I4;
    I(:,:,4) = I0;
%     I(:,:,5) = I4;
%     Ibar = mean(I(:));
%     Istd = std(I(:));
%     I = (I-Ibar)./Istd;
%     I = imresize(I,[360 480]);    
    
    % Write to disk.
    [~, filename, ext] = fileparts(info.Filename);
    imwrite(I,[imageFolder, '\',filename '.tiff'])
end
imdsTest = imageDatastore(imageFolder);
%%
reset(imdsTest)
reset(pxdsTest1)
I = read(imdsTest);
C = read(pxdsTest1);
clf
I = imresize(I,1);
L = imresize(uint8(C),5);
subplot(121);imagesc(I(:,:,1));axis image
subplot(122);imagesc(L);axis image
%%
subplot(221);imagesc(I(:,:,1));axis image
subplot(222);imagesc(I(:,:,2));axis image
subplot(223);imagesc(I(:,:,3));axis image
subplot(224);imagesc(I(:,:,4));axis image;colormap gray


%%
reset(imdsTrain1)
reset(pxdsTrainResults1)
reset(pxdsTrainResults4)
reset(pxdsTrainResults2)
imageFolder = fullfile('C:\sea urchin\Hiearhical network\data\combine\combineTrain_50');
while hasdata(pxdsTrainResults2)
    % Read an image.
    [I0,info] = read(imdsTrain1);
    [I1,info] = read(pxdsTrainResults1);
    [I4,info] = read(pxdsTrainResults4);
    [I2,info] = read(pxdsTrainResults2);
    %upsample
    I0 = double(I0(:,:,1));
    I0 = (I0-min(I0(:)))./(max(I0(:))-min(I0(:)));
    I1 = uint8(I1);I1 = (I1-min(I1(:)))./(max(I1(:))-min(I1(:)));
    I2 = imresize(uint8(I2),2);
    I2 = uint8(I2);I2 = (I2-min(I2(:)))./(max(I2(:))-min(I2(:)));
%     I4 = impyramid(double(I4),'expand');
    I4 = imresize(uint8(I4),4);
    I4 = (I4-min(I4(:)))./(max(I4(:))-min(I4(:)));
    % combine
    I = zeros(256,256,4);
    I(:,:,1)=I1;
    I(:,:,2)=I2;
    I(:,:,3)=I4;
    I(:,:,4) = I0;
%     Ibar = mean(I(:));
%     Istd = std(I(:));
%     I = (I-Ibar)./Istd;
%     I = imresize(I,[360 480]);    
    
    % Write to disk.
    [~, filename, ext] = fileparts(info.Filename);
    imwrite(I,[imageFolder, '\',filename '.tif'])
end
imdsTrain = imageDatastore(imageFolder);
%% train
numFilters = 64;
filterSize = 3;
numClasses = 2;
layers = [
    imageInputLayer([256 256 4])
    convolution2dLayer(filterSize,32,'Padding',1)
    reluLayer()
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    convolution2dLayer(1,numClasses);
    softmaxLayer()
    pixelClassificationLayer()
    ];
%% training options(look into these parameters)
options = trainingOptions('adam', ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.0005, ...
    'MaxEpochs',5, ...  
    'MiniBatchSize',8, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', '', ...
    'VerboseFrequency',2,...
    'Plots','training-progress');
%     'Momentum',0.9, ...
%% start training
pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain1);%, ...
 %   'DataAugmentation',augmenter);
doTraining = 1;
if doTraining    
    [net, info] = trainNetwork(pximds,layers,options);
else
    data = load(pretrainedSegNet);
    net = data.net;
end
%%
combinenet50 = net;
save('combinenet50.mat','combinenet50');
%%
clf
% load combinenet50
net = combinenet50;
% reset(imdsTest)
% reset(pxdsTest1)
% reset(imdsTrain1)
I = read(imdsTest);
C = semanticseg(I, net);
% B = rgb2gray(labeloverlay(I,C));
testlabel = read(pxdsTest1);
% Bb = rgb2gray(labeloverlay(I,testlabel));
L = imresize(uint8(testlabel),1);
%
L1 = uint8(C);
% I = read(imdsTrain1);
subplot(131);imagesc((I(:,:,4)));axis image

subplot(132);imagesc(L);axis image
subplot(133);imagesc(L1);axis image;
colormap gray
set(gcf,'color','white')
%%
subplot(221);imagesc(I(:,:,1));axis image
subplot(222);imagesc(I(:,:,2));axis image
subplot(223);imagesc(I(:,:,3));axis image
subplot(224);imagesc(I(:,:,4));axis image;colormap gray
%%
newimageDir = fullfile('C:\sea urchin\Hiearhical network\data\final_50');
pxdsResults= semanticseg(imdsTest, net, "WriteLocation", newimageDir,'MiniBatchSize',16);
metrics = evaluateSemanticSegmentation(pxdsResults, pxdsTest1);
% %%
% Psnr = zeros(length(pxdsResults.Files),1);
% for i = 1:length(pxdsResults.Files)
%     im1 = pxdsResults.Files(i);
%     im2 = pxdsTest1.Files(i);
%     
%     Psnr(i) = immse(imread(im1{1}),imread(im2{1}));
% end
% %%
% clf
% cdfplot(Psnr)
% xlabel('PSNR')
% ylabel('Frequency')
% %%
% Eme1 = zeros(length(pxdsResults.Files),1);
% Eme2 = zeros(length(pxdsResults.Files),1);
% for i = 1:length(pxdsResults.Files)
%     temp1 = pxdsResults.Files(i);
%     temp2 = pxdsTest1.Files(i);%imdsTrain1 pxdsTest1
%     im1 = double(imread(temp1{1}));
% %     im1(im1==1)=0;im1(im1==2)=255;
%     im2 = double(imread(temp2{1}));
%     im2(im2==0)=1; im2(im2==255)=2;
%     L = 5;
%     Eme1(i) = eme(im1,256,L);
%     Eme2(i) = eme(im2,256,L);
% %     Psnr(i) = psnr(imread(im1{1}),imread(im2{1}));
% end
% %
% Eme = Eme1 - Eme2;
% imagesc(im1)
% %
% % [f2,x2] = ecdf(Eme);
% % clf
% % plot(x2,f2*598)
% % % legend('our method','Unet')
% % xlabel('EME')
% % ylabel('Number')
% %
% Eme50 = Eme;
% %%
% load Eme5unet
% [f1,x1] = ecdf(abs(Eme50));
% [f2,x2] = ecdf(abs(Eme));
% 
% clf
% plot(x1,f1*598)
% hold on
% plot(x2,f2*598)
% legend('our method','Unet')
% xlabel('EME_{thr}')
% ylabel('# of images <= EME_{thr}')
% title('L = 5')
% set(gcf,'color','white')
% %%
% clf
% histogram(Eme50)
% hold on
% histogram(Eme)
% legend('our method','Unet')
% title('L=30')
% set(gcf,'color','white')
% %%
% %%
% reset(pxdsResults)
% reset(pxdsTest1)
% Psnr = zeros(length(pxdsResults.Files),1);
% for i = 1:1%length(pxdsResults.Files)
%     im1 = (read(pxdsResults));
%     im2 = (read(pxdsTest1));
%     [score,precision,recall] = bfscore((im1),(im2));
% end
% plot(precision,recall)
%%
% [score,precision,recall] = bfscore(pxdsResults,pxdsTest1);
% [ap, recall, precision] = evaluateDetectionPrecision(cell2table(pxdsResults.Files), cell2table(pxdsTest1.Files));