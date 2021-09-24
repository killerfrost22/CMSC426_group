clc
clear all
close all
%% hyperparermeters
thresh = .000008;
k = 10;
%% training
% read images
path = dir('./train_images/*.jpg');
filenames={path.name};
%filenames=filenames(1:2);
num_train_images = length(filenames);
orange = [];
curr = zeros(1,3);
for i = 1:num_train_images
    name_full = filenames{1,i};
    name = name_full(1:length(name_full)-4);
    img = imread(strcat('./train_images/', name,'.jpg'));
    mask = imread(strcat('./train_images/masks/', name,'-mask.bmp'));
    [rows, cols, colorChannels] = size(img);
    
    for i=1:rows
    for j=1:cols
        if mask(i,j) == 255
            curr(1,1) = img(i,j,1);
            curr(1,2) = img(i,j,2);
            curr(1,3) = img(i,j,3);
            orange = vertcat(orange,double(curr));
        end
    end
    end
    
end
x = orange;
[mu, sigma] = trainGMM(x, k);
plotGMM(mu, sigma, 1, x);
%% testing
path = dir('./test_images/*.jpg');
test_filenames={path.name};
num_of_test_img = length(test_filenames);

 for i = 1:num_of_test_img
    currImg = imread(strcat('./test_images/', test_filenames{i}));
    x = currImg;    
    cluster = testGMM(x, mu, sigma, thresh, k);    
    % Measure Depth
    d = measureDepth(cluster');
    figure();
    imshow(cluster'); 
    disp("Distance of Ball Figure "+i+": "+d);    
 end
 

