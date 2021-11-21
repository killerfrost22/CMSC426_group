% MyRotobrush.m  - UMD CMSC426, Fall 2018
% This is the main script of your rotobrush project.
% We've included an outline of what you should be doing, and some helful visualizations.
% However many of the most important functions are left for you to implement.
% Feel free to modify this code as you see fit.

clear;

% True if mask comes from user input
userInput = false;

% Some parameters you need to tune:
WindowWidth = 50;  
ProbMaskThreshold = 0.5; 
NumWindows= 50; 
BoundaryWidth = 2;

% Delete any previous output:
foutput = '../output';
delete(strcat(foutput, '/*'));

% Load images:
fpath = '../input';
files = dir(fullfile(fpath, '*.jpg'));
imageNames = zeros(length(files),1);
images = cell(length(files),1);

for i=1:length(files)
    imageNames(i) = str2double(strtok(files(i).name,'.jpg'));
end

imageNames = sort(imageNames);
imageNames = num2str(imageNames);
imageNames = strcat(imageNames, '.jpg');

for i=1:length(files)
    images{i} = im2double(imread(fullfile(fpath, strip(imageNames(i,:)))));
end

% NOTE: to save time during development, you should save/load your mask rather than use ROIPoly every time.
if userInput
    mask = roipoly(images{1});
else
    mask = imread(strcat(fpath, '/mask1.png'));
end
mask = logical(mask(:,:,1));

figure('Name', 'Initial mask boundary');
imshow(imoverlay(images{1}, boundarymask(mask,8),'red'));
set(gca,'position',[0 0 1 1],'units','normalized')
F = getframe(gcf);
[I,~] = frame2im(F);
%imwrite(I, fullfile(fpath, strip(imageNames(1,:))));
outputVideo = VideoWriter(fullfile(strcat(foutput, '/video.mp4')),'MPEG-4');
open(outputVideo);
writeVideo(outputVideo,I);

% Sample local windows and initialize shape+color models:
[mask_outline, LocalWindows] = initLocalWindows(images{1},mask,NumWindows,WindowWidth,false);

% Show initial local windows:
figure('Name', 'Local Windows');
imshow(images{1})
hold on
showLocalWindows(LocalWindows,WindowWidth,'r.');
hold off
set(gca,'position',[0 0 1 1],'units','normalized')
F = getframe(gcf);
[I,~] = frame2im(F);

ColorModels = ...
    initColorModels(images{1},mask,mask_outline,LocalWindows,BoundaryWidth,WindowWidth);

% You should set these parameters yourself:
fcutoff = 0.75;
SigmaMin = 2;
SigmaMax = WindowWidth;
R = 2;
A = (SigmaMax-SigmaMin)/(1-fcutoff)^R;
ShapeConfidences = ...
    initShapeConfidences(LocalWindows,ColorModels,...
    WindowWidth, SigmaMin, A, fcutoff, R);
ShapeConfidences.prob_mask = double(mask);

figure('Name', 'Color confidences');
showColorConfidences(images{1},mask_outline,ColorModels.Confidences,LocalWindows,WindowWidth);

%%% MAIN LOOP %%%
% Process each frame in the video.
for prev=1:(length(files)-1)
    curr = prev+1;
    fprintf('Current frame: %i\n', curr)
    
    %%% Global affine transform between previous and current frames:
    [warpedFrame, warpedMask, warpedMaskOutline, warpedLocalWindows] = calculateGlobalAffine(images{prev}, images{curr}, mask, mask_outline, LocalWindows);
    
    %%% Calculate and apply local warping based on optical flow:
    NewLocalWindows = ...
        localFlowWarp(warpedFrame, images{curr}, warpedLocalWindows,warpedMask,WindowWidth);
    
    % Show windows before and after optical flow-based warp:

    figure('Name', 'Effect of optical flow');
    imshow(images{curr});
    hold on
    showLocalWindows(warpedLocalWindows,WindowWidth,'r.');
    showLocalWindows(NewLocalWindows,WindowWidth,'b.');
    hold off
    
    
    %%% UPDATE SHAPE AND COLOR MODELS:
    % This is where most things happen.
    % Feel free to redefine this as several different functions if you prefer.
    [ ...
        mask, ...
        LocalWindows, ...
        ColorModels, ...
        ShapeConfidences, ...
    ] = ...
    updateModels(...
        NewLocalWindows, ...
        LocalWindows, ...
        images{curr}, ...
        warpedMask, ...
        warpedMaskOutline, ...
        WindowWidth, ...
        ColorModels, ...
        ShapeConfidences, ...
        ProbMaskThreshold, ...
        fcutoff, ...
        SigmaMin, ...
        R, ...
        A ...
    );

    mask_outline = bwperim(mask,4);

    % Write video frame:
    %figure('Name', sprintf('Output frame %d', curr));
    imshow(imoverlay(images{curr}, boundarymask(mask,8), 'red'));
    set(gcf,'name', sprintf('Output frame %d', curr));
    set(gca,'position',[0 0 1 1],'units','normalized')
    F = getframe(gcf);
    [I,~] = frame2im(F);
    imwrite(I, fullfile(foutput, strip(imageNames(curr,:))));
    writeVideo(outputVideo,I);

end
close(outputVideo);

%figure();
%imshow(cat(3,ones(2000,5000),zeros(2000,5000,2)));
