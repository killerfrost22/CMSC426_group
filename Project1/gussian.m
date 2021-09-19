clear 
clc 
close all 

%% Train Constant 
Train = false

%% Load Data
selector = strcat('train_images', '/*.jpg');
path = dir(selector);
imagePath = strcat('train_images/', path.name);
saveFile = 'data/singleGaussian.mat'

%% Process Data 
for number = 1: size(imagePath)
    disp(imagePath)
    filename = fullfile(path(number).folder, path(number).name);
    I = imread(filename);
    imshow(I);
    
    % Form Mask
    BW = uint8(roipoly(I));
    
    %Get Dimensions
    picSize = size(I)
    width = sz(1)
    height = sz(2)
    
    R = rgbs(:,:,1);
    G = rgbs(:,:,2);
    B = rgbs(:,:,3);
    
    % Masked image for visualization purposes
    maskedI = uint8(zeros(size(I))); 
    maskedI(:,:,1) = r .* BW;
    maskedI(:,:,2) = g .* BW;
    maskedI(:,:,3) = b .* BW;

    for i = size(b,1)
        for j = 1: size(n,2)
            if b(i,j) > 0 
                area = area + 1; 
                mean = mean + [double(R(1,j)); double(B(i,j))];
            end
        end
    end
    
    s = s + area; 
    
                
            
end