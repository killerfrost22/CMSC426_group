clear
clc
close all

%% Train Constant

%Change to false to load from singleGaussModel.mat
TRAIN = false;

%% Initialize

%Note, this folder can be made into more images. This file is scalable.
selector = strcat('train_images', '/*.jpg');
path = dir(selector);
imgN = length(path);
saveFileName = 'singleGaussModel.mat';

%% Grab All Orange Pixels from Training Data
if(TRAIN)
    orange = [];
    % For each image
    for i = 1:imgN
        disp(path)
        imgPath = fullfile(path(i).folder, path(i).name);
        I = imread(imgPath);
        imshow(I);

        % Get Dims
        sz = size(I);
        width = sz(1);
        height = sz(2);

        % Form Mask
        BW = uint8(roipoly(I));

        % Get RGB values for image
        r = I(:,:,1);
        g = I(:,:,2);
        b = I(:,:,3);
        
        % Masked image for visualization purposes
        maskedI = uint8(zeros(size(I))); 
        maskedI(:,:,1) = r .* BW;
        maskedI(:,:,2) = g .* BW;
        maskedI(:,:,3) = b .* BW;

        %imshow(maskedI);

        % Look at every pixel. If it's 1 in the roipoly image, it's orange.
        % Add it to the running list of orange pixels.
        nO = 0;
        maskedI = double(maskedI);
        for x = 1:width
            for y = 1:height
                if BW(x,y) == 1
                    %Add to list of oranges pixels
                    orange = [orange reshape(maskedI(x,y,:),3,1)];
                    %Increment how many orange pixels counted
                    nO = nO+1;
                end
            end
        end
    end

    %% Calculate Mean and Covariance
    %Go over ALL orange pixels seen to establish mean and cov
    mu = double(zeros(3,1));

    for i=1:nO
       mu = mu + orange(:,i);
    end
    mu = mu/nO;
    
    disp("Empirical Mean")
    disp(mu)

    sigma = double(zeros(3,3));
    
    for i=1:nO
       a = orange(:,i)-mu;
       sigma = sigma + (a * a');
    end
    sigma = sigma/nO;
    
    disp("Empirical Covariance");
    disp(sigma);

    %% Save Data

    %Save mean and cov
    save(saveFileName, 'mu', 'sigma');
else
    load(saveFileName, 'mu', 'sigma');
end % END IF STATEMENT

%% Predict
threshold = .0000004;
prior = .5;

selector = strcat('test_images', '/*.jpg');
path = dir(selector);
imgN = length(path);

for i = 1:imgN
        disp("Image")
        disp(i)
        %disp(path)
        imgPath = fullfile(path(i).folder, path(i).name);
        I = imread(imgPath);
        %imshow(I);
        
        % Get Dims
        sz = size(I);
        height = sz(2);
        width = sz(1);
        
        %Init prediction image to all black
        prediction = uint8(zeros(width,height));
        
        %For each pixel in the test image
        for x=1:width
            for y=1:height
                %Form RGB value
                ex = [double(I(x,y,1)); double(I(x,y,2)); double(I(x,y,3))];
                l = likelihood(ex, sigma, mu, 3);
                p = prob(l,prior);
                %Does our model consider it orange?
                if p >= threshold
                    %If so, color it white in the prediction
                    prediction(x,y) = 1;
                end
            end
        end
        figure('Name',strcat('Test Image ',num2str(i)));
        imshow(prediction,[]);
end

%% Helpers

%Bayes Rule (aka Posterior)
function p = prob(likelihood, prior)
    p = likelihood * prior;
end

function l = likelihood(x,sigma,mu,N)
    a = 1/(sqrt((2*pi)^N*det(sigma)));
    b = exp(-.5*(x-mu)'*(sigma\(x-mu)));
    l = a*b;
end