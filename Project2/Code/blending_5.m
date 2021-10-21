% location = "./"
% directory = dir(location)
% disp(directory)

% NOTE:
% For file loading to work correctly,
% the "Current Folder" chosen on left hand side must be located at root
% location of directory

location = ".\Project2\Images\train_images\Set3\";
% location = "/Users/gyq888/Desktop/Project2/Images/Set1/";

directory = dir(location + '*.jpg')
num_images=size(directory,1)

% Initialize all the transforms to the identity matrix. Note that the
% projective transform is used here because the building images are fairly
% close to the camera. Had the scene been captured from a further distance,
% an affine transform would suffice.

tforms(num_images) = projective2d(eye(3));

% Initialize variable to hold image sizes.
imageSize = zeros(num_images,2);

% Iterate over remaining image pairs
for n = 2:num_images   
    curr_path = directory(n-1);
    folderchar = curr_path.folder;
    folder = convertCharsToStrings(folderchar) + "\";
    filechar = curr_path.name;
    file = convertCharsToStrings(filechar);
    path = folder + file
    img1 = imread(path);
    
    curr_path = directory(n);
    folderchar = curr_path.folder;
    folder = convertCharsToStrings(folderchar) + "\";
    filechar = curr_path.name;
    file = convertCharsToStrings(filechar);
    path = folder + file
    img2 = imread(path);
    
%     imshow(img1);
%     figure;
%     imshow(img2);
    
    gray1 = rgb2gray(img1);
    gray2 = rgb2gray(img2);
    
    corners1 = cornermetric(gray1);
    corners2 = cornermetric(gray2);
    
    max1 = imregionalmax(corners1);
    max2 = imregionalmax(corners2);
    
    n_best = 250; 
    [X1, Y1] = ANMS(n_best, gray1, corners1, max1);
    [X2, Y2] = ANMS(n_best, gray2, corners2, max2);
%     displayANMS(img1, Y1, X1);
%     displayANMS(img2, Y2, X2);
    
    [d1, filteredX1, filteredY1] = feature_descriptors(gray1, X1, Y1);
    [d2, filteredX2, filteredY2] = feature_descriptors(gray2, X2, Y2);
    displayANMS(img1, filteredX1, filteredY1);
    displayANMS(img2, filteredX2, filteredY2);
    
    match_threshold = 0.5;
    N_max = 25
    RANSAC_thresh = 0.00001
    
    [MATCHEDXY, filtered, matchedp1X, matchedp1Y, matchedp2X, matchedp2Y] = getMatchedPoints(img1, img2, match_threshold, ...
    d1, d2, filteredX1, filteredY1, filteredX2, filteredY2);
    
    figure
    hprevImage = showMatchedFeatures(img1, img2, [matchedp1X, matchedp1Y], [matchedp2X, matchedp2Y], 'montage')
    disp("")
    
    [INLIERSp1X, INLIERSp1Y, INLIERSp2X, INLIERSp2Y] = RANSAC(N_max, RANSAC_thresh, ...
        matchedp1X, matchedp1Y, matchedp2X, matchedp2Y);
    
    figure
    hImage = showMatchedFeatures(img1, img2, [INLIERSp1X, INLIERSp1Y], [INLIERSp2X, INLIERSp2Y], 'montage')
    disp("")
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    imageSize(n,:) = size(gray1);
    
    % Estimate the transformation between I(n) and I(n-1).
    H12 = est_homography(INLIERSp2X,INLIERSp2Y, INLIERSp1X,INLIERSp1Y);
    H21 = est_homography(INLIERSp1X,INLIERSp1Y, INLIERSp2X,INLIERSp2Y);
    
    warp12 = projective2d(H12');
    warp21 = projective2d(H21');

    tforms(n) = warp21;
    
    % Compute T(n) * T(n-1) * ... * T(1)
    tforms(n).T = tforms(n).T * tforms(n-1).T; 
end


% Compute the output limits for each transform.
for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), ...
        [1 imageSize(i,2)], [1 imageSize(i,1)]);    
    disp("")
end

avgXLim = mean(xlim, 2); [~,idx] = sort(avgXLim);
% avgYLim = mean(ylim, 2);
% [~,idx] = sort(avgYLim);
centerIdx = floor((numel(tforms)+1)/2);
centerImageIdx = idx(centerIdx);

% centerImageIdx = 1;

Tinv = invert(tforms(centerImageIdx));
for i = 1:numel(tforms)    
    tforms(i).T = tforms(i).T * Tinv.T;
end

for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), ...
        [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

maxImageSize = max(imageSize);

% Find the minimum and maximum output limits. 
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

% Width and height of panorama.
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Initialize the "empty" panorama.
panorama = zeros([height width 3], 'like', img1);

blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');  

% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% Create the panorama.
for i = 1:num_images
    
    curr_path = directory(i);
    folderchar = curr_path.folder;
    folder = convertCharsToStrings(folderchar) + "/"
    filechar = curr_path.name;
    file = convertCharsToStrings(filechar)
    path = folder + file
    
    I = imread(path);

    imshow(I);
    
%     I = readimage(buildingScene, i);   
   
    % Transform I into the panorama.
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);

    imshow(warpedImage);
                  
    % Generate a binary mask.    
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);

    imshow(mask);
    
    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage, mask);

    imshow(panorama);

    disp("")
end

figure
imshow(panorama)
disp("")

function [descriptors, X, Y] = feature_descriptors(img_grayscale, x, y)
    [~, n_best] = size(x);
    [ x_size, y_size] = size(img_grayscale);
    descriptors = []; X = []; Y = [];
    
    
    
    for i = 1:n_best
        % Patch is of size 41x41, so point is the actual center
        
        x_i = x(i);
        y_i = y(i);
        
        dim = 20;

        if (((x_i-dim-1) >= 1) &&  ((y_i-dim-1)>=1) && ((x_i + dim) <= x_size) && ((y_i + dim) <= y_size))
            
            patch = img_grayscale(x_i - dim - 1:x_i + dim, y_i - dim - 1:y_i + dim);
            
%             imshow(patch)
%             figure(10,10); imshow(blurred)

%             avg5 = ones(5)/9;
%             C = conv2(avg5,blurred )
            
%             H = fspecial('motion', 10, 45);
                
            % h = fspecial('gaussian',hsize,sigma) returns 
            % a rotationally symmetric Gaussian lowpass filter of size hsize 
            % with standard deviation sigma. Not recommended.
            
            
            
            hsize = 10; sigma = 0.01;
%             H = fspecial('gaussian',hsize,sigma);
%             blurred = imfilter(patch,H,'replicate'); 
%             imshow(blurred);

            blurred = imgaussfilt(patch, sigma);

%             resized = blurred(1:5:end, 1:5:end);
            resized = imresize(blurred, [8 8]);
            
            reshaped = double(reshape(resized, [64, 1]));
            
            meanlimit = mean(reshaped);
            
            std_dev = std(reshaped);
            mean_reshaped = mean(reshaped);
            standardized = (reshaped - mean_reshaped) ./ std_dev;
            
%             if (meanlimit <= 120 || meanlimit >= 150)
%                 standardized = 99999*ones(64,1);
% %                 standardized = 500 + (999999-500) .* rand(64,1);
%             end
            
            
            descriptors = [descriptors standardized];
            X = [X y_i];
            Y = [Y x_i];
            disp("")
        end
    end
    disp("")
end


function [INLIERSp1X, INLIERSp1Y, INLIERSp2X, INLIERSp2Y] = RANSAC(N_max, thresh, match1X, match1Y, match2X, match2Y)
    
    total = size(match1X, 1); inliers_count = 0; iter = 0;
    INLIERSXY = [];
    ssds = [];
    
    random_size = 4
    percentage = 0;
    while (iter < N_max && (percentage < 0.50) && inliers_count < 10)

        random_i = randperm(total, random_size);

        % 1. Select four pairs of matched pixels (at random), 1<=i<=4, p_1i/p_2i from images 1/2.
        tempX1 = match1X(random_i); tempY1 = match1Y(random_i);
        tempX2 = match2X(random_i); tempY2 = match2Y(random_i);
        
%         if (ismember(346,tempX1) || ismember(312,tempX1) || ismember(407,tempX1))
%             disp("")
%         end
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % 2. Compute the homography matrix from the four feature pairs using est_homography

        % destination, source
        H = est_homography(tempX1,tempY1, tempX2,tempY2);
        
        % 3. Compute inliers where SSD(Hp_1, p_2) < threshold
        for i=1:random_size

            p1 = [tempX1(i); tempY1(i)];
            [Hp2X, Hp2Y] = apply_homography(H, [tempX2(i)], [tempY2(i)]);
            Hp2 = [Hp2X; Hp2Y];            
            
            X = Hp2 - p1;
            ssd = sum(X(:).^2);
            
            if (tempX1(i) == 346 || tempX1(i)==312 || tempX1(i)==407 ...
                || tempX1(i)==184 || tempX1(i)==202 || tempX1(i)==394 ...
                || tempX1(i)==272 || tempX2(i)==777 || tempX1(i)==777 ...
                || tempY1(i)==777 || tempY2(i)==777 ...
                || tempX1(i)==168 || tempX2(i)==168 ...
                || tempX1(i)==190 || tempX2(i)==190)
                disp("wtf")
                disp("")
            end

            if (ssd < thresh)
                
                if (tempX1(i) == 346 || tempX1(i)==312 || tempX1(i)==407 ...
                || tempX1(i)==184 || tempX1(i)==202 || tempX1(i)==394 ...
                || tempX1(i)==272 || tempX2(i)==777 || tempX1(i)==777 ...
                || tempY1(i)==777 || tempY2(i)==777 ...
                || tempX1(i)==168 || tempX2(i)==168 ...
                || tempX1(i)==190 || tempX2(i)==190)
                    disp("wtf")
                    disp("")
                end

                INLIERSXY = [INLIERSXY; [tempX1(i), tempY1(i), tempX2(i), tempY2(i)]];
                INLIERSXY = unique(INLIERSXY, 'rows', 'stable');
                inliers_count = size(INLIERSXY,1);
                
                if (inliers_count >= 10)
                    break;
                end
                
            end

        end
        
        last_percentage = percentage;
        percentage = (inliers_count/total);
        if (last_percentage < percentage)
%             percentage
            fprintf("iter: %d, ssd: %f, percentage: %f\n", iter, ssd, percentage)
        end

        % 4. Repeat the last 3 steps until you reach N_max iters or 90% of inliers.
        iter = iter + 1;
        % 5. Keep largest set of inliers.
    end

    % 6. Recompute least-square H on all inliers.    
%     H_ls = est_homography(INLIERSXY(:,3),INLIERSXY(:,4), INLIERSXY(:,1),INLIERSXY(:,2));

%     filteredInliers = [];
%     for i=1:size(INLIERSXY, 1)
%         for j=1:size(INLIERSXY, 1)
%             
%             if (i == j)
%                 continue;
%             end
%             
%             DUPLICATES_PRESENT = false;
%             if ((INLIERSXY(i,1) ==  INLIERSXY(j,1) && INLIERSXY(i,2) == INLIERSXY(j,2)) ...
%                     || (INLIERSXY(i,3) ==  INLIERSXY(j,3) && INLIERSXY(i,4) == INLIERSXY(j,4)))
%                 DUPLICATES_PRESENT = true;
%                 break;
%             end
%         end
%         if not(DUPLICATES_PRESENT)
%             rowToAdd = INLIERSXY(i,:);
%             filteredInliers = [filteredInliers; rowToAdd];
%         end
%     end
%     
%     if (size(filteredInliers,1) > 0)
%         INLIERSp1X = filteredInliers(:,1);
%         INLIERSp1Y = filteredInliers(:,2);
%         INLIERSp2X = filteredInliers(:,3);
%         INLIERSp2Y = filteredInliers(:,4);
%     else
        INLIERSp1X = INLIERSXY(:,1);
        INLIERSp1Y = INLIERSXY(:,2);
        INLIERSp2X = INLIERSXY(:,3);
        INLIERSp2Y = INLIERSXY(:,4);    
%     end

    iter
    percentage
    
    fprintf("end RANSAC");
end


function [MATCHEDXY, filtered, matchedp1X, matchedp1Y, matchedp2X, matchedp2Y] = getMatchedPoints(img1, img2, thresh, d1, d2, p1X, p1Y, p2X, p2Y)
    matchedp1X = [];matchedp1Y = [];matchedp2X = [];matchedp2Y = [];
    d1size = length(d1(1,:));d2size = length(d2(2,:));
    
    ratios = [];
    mindist = [];
    
    for i = 1:d1size
        sumSquare = zeros(1,d2size);
        for j = 1:d2size
            sumSquare(j) = sum((d1(:,i) - d2(:,j)).^2);
        end
        [sortedDist, I] = sort(sumSquare);
        ratio = sortedDist(1)/sortedDist(2);
        
%         if (sortedDist(1) <= 0.8)
%             continue;
%         end
%         0.05 < ratio && 
        if (ratio < thresh)
            
            mindist = [mindist sortedDist(1)];
            ratios = [ratios; ratio];
            
            matchedp1X = [matchedp1X; p1X(i)];
            
            if (ismember(190, matchedp1X))
                disp("")
            end
            
            matchedp1Y = [matchedp1Y; p1Y(i)];

            matchedp2X = [matchedp2X; p2X(I(1))];
            matchedp2Y = [matchedp2Y; p2Y(I(1))];
        end
    end
    
    showMatchedFeatures(img1, img2, [matchedp1X, matchedp1Y], [matchedp2X, matchedp2Y], 'montage')
    
    [sortedMinDist, ~] = sort(mindist)
    ratios
    disp("")

    
    MATCHEDXY = [matchedp1X, matchedp1Y, matchedp2X, matchedp2Y];
    filtered = [];
    for i=1:size(MATCHEDXY, 1)
        for j=1:size(MATCHEDXY, 1)
            
            if (i == j)
                continue;
            end
            
            DUPLICATES_PRESENT = false;
            if ((MATCHEDXY(i,1) ==  MATCHEDXY(j,1) && MATCHEDXY(i,2) == MATCHEDXY(j,2)) ...
                    || (MATCHEDXY(i,3) ==  MATCHEDXY(j,3) && MATCHEDXY(i,4) == MATCHEDXY(j,4)))
                DUPLICATES_PRESENT = true;
                break;
            end
        end
        if not(DUPLICATES_PRESENT)
            rowToAdd = MATCHEDXY(i,:);
            filtered = [filtered; rowToAdd];
        end
    end
    
    if (size(filtered,1) > 0)
        matchedp1X = filtered(:,1);
        matchedp1Y = filtered(:,2);
        matchedp2X = filtered(:,3);
        matchedp2Y = filtered(:,4);
    else
        matchedp1X = MATCHEDXY(:,1);
        matchedp1Y = MATCHEDXY(:,2);
        matchedp2X = MATCHEDXY(:,3);
        matchedp2Y = MATCHEDXY(:,4);    
    end
    
end

function [x_best, y_best] = ANMS(n_best, gray_img, corners, maxima)
    
    NStrong = 0;
    [sizeX, sizeY] = size(gray_img);
    x = []; y = [];

    for x_i = 1:sizeX
        for y_i = 1:sizeY
            if maxima(x_i,y_i) == 1
                NStrong = NStrong + 1;
                x = [x x_i];
                y = [y y_i]; 
            end
        end
    end
    
    r = Inf(1, NStrong);
    for i = 1:NStrong
        for j = 1:NStrong
%             if img_corners(y(j), x(j)) > img_corners(y(i), x(i))
            if corners(x(j), y(j)) > corners(x(i), y(i))
                ED = ((x(j) - x(i))^2) + ((y(j) - y(i))^2);
                if ED < r(i)
                    r(i) = ED;
                end
            end
        end
    end
    
    x_best = zeros(1, n_best);
    y_best = zeros(1, n_best);    
    [~, I] = sort(r, 'descend');
    
    for i = 1:n_best
        x_best(i) = x(I(i));
        y_best(i) = y(I(i));
    end
end

function displayANMS(img, x_best, y_best)
    figure;
    imshow(img);
    hold on
%      plot(x_best, y_best, 'Color', 'r', 'Marker','.', 'LineStyle','-')
    plot(x_best, y_best, 'r.')
    hold off
end

function H = est_homography(X,Y,x,y)
    % H = est_homography(X,Y,x,y)
    % Compute the homography matrix from source(x,y) to destination(X,Y)
    %
    %    X,Y are coordinates of destination points
    %    x,y are coordinates of source points
    %    X/Y/x/y , each is a vector of n*1, n>= 4
    %
    %    H is the homography output 3x3
    %   (X,Y, 1)^T ~ H (x, y, 1)^T
    
    A = zeros(length(x(:))*2,9);
    
    for i = 1:length(x(:))
     a = [x(i),y(i),1];
     b = [0 0 0];
     c = [X(i);Y(i)];
     d = -c*a;
     A((i-1)*2+1:(i-1)*2+2,1:9) = [[a b;b a] d];
    end
    
    [U S V] = svd(A);
    h = V(:,9);
    H = reshape(h,3,3)';
end

function [X, Y] = apply_homography(H, x, y)
    % [X, Y] = apply_homography(H, x, y)
    % Use homogrphay matrix H to compute position (x,y) in the source image to
    % the position (X,Y) in the destination image
    %
    % Input
    %   H : 3*3 homography matrix, refer to setup_homography
    %   x : the column coords vector, n*1, in the source image
    %   y : the column coords vector, n*1, in the source image
    % Output
    %   X : the column coords vector, n*1, in the destination image
    %   Y : the column coords vector, n*1, in the destination image
    
    p1 = [x'; y'; ones(1, length(x))];
    q1 = H*p1;
    q1 = q1./[q1(3, :); q1(3,:); q1(3, :)];
    
    X = q1(1,:)';
    Y = q1(2, :)';
end

function hImage = showMatchedFeatures(I1, I2, matchedPoints1, matchedPoints2, varargin)
%showMatchedFeatures Display corresponding feature points.
%  showMatchedFeatures(I1, I2, matchedPoints1, matchedPoints2) displays a
%  falsecolor overlay of images I1 and I2, with a color-coded plot of the
%  corresponding points connected by a line. matchedPoints1 and
%  matchedPoints2 are the coordinates of corresponding points in I1 and
%  I2. Points can be an M-by-2 matrix of [x y] coordinates, a SURFPoints
%  an MSERRegions, a cornerPoints, or a BRISKPoints object.
%
%  showMatchedFeatures(I1, I2, matchedPoints1, matchedPoints2, method)
%  displays images I1 and I2 using the visualization style specified by
%  method. Values of method can be:
%
%    'falsecolor' : Overlay the images by creating a composite red-cyan 
%                   image showing I1 as red and I2 as cyan.
%    'blend'      : Overlay I1 and I2 using alpha blending.
%    'montage'    : Place I1 and I2 next to each other in the same image.
%
%    Default: 'falsecolor'
%
%  hImage = showMatchedFeatures(...) returns the handle to the image object
%  returned by showMatchedFeatures.
%
%  showMatchedFeatures(...,Name,Value) specifies additional name-value pair 
%  arguments described below:
%
%  'PlotOptions'  Specify custom plot options in a cell array containing 
%                 three string values, {MarkerStyle1, MarkerStyle2, LineStyle},
%                 corresponding to marker specification in I1, marker 
%                 specification in I2, and line style and color. Each of 
%                 the specifiers is defined by the <a href="matlab:doc('linespec')">LineSpec</a> of PLOT function.
%
%                 Default: {'ro','g+','y-'}
%
%   'Parent'      Specify an output axes for displaying the visualization.
%
%  Class Support
%  -------------
%  I1 and I2 are numeric arrays.
%
%  Example 1
%  ---------
%  % Use Harris features to find corresponding points between two images.
%  I1 = rgb2gray(imread('parkinglot_left.png'));
%  I2 = rgb2gray(imread('parkinglot_right.png'));
%
%  points1 = detectHarrisFeatures(I1);
%  points2 = detectHarrisFeatures(I2);
%   
%  [f1, vpts1] = extractFeatures(I1, points1);
%  [f2, vpts2] = extractFeatures(I2, points2);
%
%  indexPairs = matchFeatures(f1, f2) ;
%  matchedPoints1 = vpts1(indexPairs(1:20, 1));
%  matchedPoints2 = vpts2(indexPairs(1:20, 2));
%
%  % Visualize putative matches
%  figure; showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2,'montage');
%
%  title('Putative point matches');
%  legend('matchedPts1','matchedPts2');
%
%  Example 2
%  ---------
%  % Use SURF features to find corresponding points between two images
%  % rotated and scaled with respect to each other
%  I1 = imread('cameraman.tif');
%  I2 = imresize(imrotate(I1,-20), 1.2);
% 
%  points1 = detectSURFFeatures(I1);
%  points2 = detectSURFFeatures(I2);
% 
%  [f1, vpts1] = extractFeatures(I1, points1);
%  [f2, vpts2] = extractFeatures(I2, points2);
%         
%  indexPairs = matchFeatures(f1, f2) ;
%  matchedPoints1 = vpts1(indexPairs(:, 1));
%  matchedPoints2 = vpts2(indexPairs(:, 2));
%
%  % Visualize putative matches
%  figure; showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2);
%
%  title('Putative point matches');
%  legend('matchedPts1','matchedPts2');
%
% See also matchFeatures, estimateGeometricTransform, imshowpair,
%     legend, SURFPoints, MSERRegions, cornerPoints

% Copyright 2011 The MathWorks, Inc.

narginchk(4,7);
 
[matchedPoints1, matchedPoints2, method, lineSpec, hAxes] = ...
    parseInputs(I1, I2, matchedPoints1, matchedPoints2, varargin{:});

% pad the smaller image
paddedSize = [max(size(I1,1), size(I2,1)), max(size(I1,2), size(I2,2))];
I1pad = [paddedSize(1) - size(I1,1), paddedSize(2) - size(I1,2)];
I2pad = [paddedSize(1) - size(I2,1), paddedSize(2) - size(I2,2)];
I1pre = round(I1pad/2);
I2pre = round(I2pad/2);
I1 = padarray(I1, I1pre, 0, 'pre');
I2 = padarray(I2, I2pre, 0, 'pre');
I1 = padarray(I1, I1pad-I1pre, 0, 'post');
I2 = padarray(I2, I2pad-I2pre, 0, 'post');

switch lower(method)
    case {'falsecolor'}
        imgOverlay = imfuse(I1, I2);
        % create red-cyan image instead of the imfuse default
        imgOverlay(:,:,1) = imgOverlay(:,:,2);
        imgOverlay(:,:,2) = imgOverlay(:,:,3);
    case {'blend'}
        imgOverlay = imfuse(I1,I2,'blend');        
    case {'montage'}
        imgOverlay = imfuse(I1,I2,'montage');
end

% Display the composite image
if nargout > 0
    hImage = imshow(imgOverlay, 'Parent', hAxes);
else
    imshow(imgOverlay, 'Parent', hAxes);
end

holdState = get(hAxes,'NextPlot'); % store the state for 'hold' before changing it
set(hAxes, 'NextPlot', 'add');

%============
% Plot points
%============
% Calculate the offsets needed to adjust plot after images were fused
offset1 = fliplr(I1pre);
offset2 = fliplr(I2pre);
if strcmp(method, 'montage')
    offset2 = offset2 + fliplr([0 size(I1,2)]);
end

matchedPoints1 = bsxfun(@plus, matchedPoints1, offset1);
matchedPoints2 = bsxfun(@plus, matchedPoints2, offset2);

if ~isempty(lineSpec{1})
    plot(hAxes, matchedPoints1(:,1), matchedPoints1(:,2), lineSpec{1}); % marker 1
end
if ~isempty(lineSpec{2})
    plot(hAxes, matchedPoints2(:,1), matchedPoints2(:,2), lineSpec{2}); % marker 2
end

% Plot by using a single line object with line segments broken by using
% NaNs. This is more efficient and makes it easier to customize the lines.
lineX = [matchedPoints1(:,1)'; matchedPoints2(:,1)'];
numPts = numel(lineX);
lineX = [lineX; NaN(1,numPts/2)];

lineY = [matchedPoints1(:,2)'; matchedPoints2(:,2)'];
lineY = [lineY; NaN(1,numPts/2)];

plot(hAxes, lineX(:), lineY(:), lineSpec{3}); % line

set(hAxes, 'NextPlot', holdState); % restore the hold state

drawnow();
end

%==========================================================================
% Input parser
%==========================================================================
function [matchedPoints1, matchedPoints2, method, lineSpec, hAxes] = ...
    parseInputs(I1, I2, matchedPoints1, matchedPoints2, varargin)

% do only basic image validation; let padarray and imfuse take care of 
% the rest
validateattributes(I1,{'numeric','logical'},{'real','nonsparse',...
    'nonempty'},mfilename,'I1',1)
validateattributes(I2,{'numeric','logical'},{'real','nonsparse',...
    'nonempty'},mfilename,'I2',2)

matchedPoints1 = parsePoints(matchedPoints1, 1);
matchedPoints2 = parsePoints(matchedPoints2, 2);

if size(matchedPoints1,1) ~= size(matchedPoints2,1)
    error(message('vision:showMatchedFeatures:numPtsMustMatch'));
end

% Process the rest of inputs
parser = inputParser;
parser.FunctionName  = mfilename;

parser.addOptional('Method', 'falsecolor', @checkMethod);
parser.addParameter('PlotOptions', {'ro','g+','y-'}, @checkPlotOptions);
parser.addParameter('Parent', [], ...
    @vision.internal.inputValidation.validateAxesHandle);

% Parse inputs
parser.parse(varargin{:});

% Calling validatestring again permits easy handling of partial string matches
method = validatestring(parser.Results.Method,...
    {'falsecolor','blend','montage'},mfilename,'Method');

lineSpec = parser.Results.PlotOptions;

hAxes = newplot(parser.Results.Parent);
end

%==========================================================================
function points=parsePoints(points, ptsInputNumber)

fcnInputVarNumber = 2 + ptsInputNumber; 
varName = ['matchedPoints', num2str(ptsInputNumber)];

if ~isa(points, 'vision.internal.FeaturePoints') && ~isa(points, 'MSERRegions')
    validateattributes(points,{'int16', 'uint16', 'int32', 'uint32', ...
        'single', 'double'}, {'2d', 'nonsparse', 'real', 'size', [NaN 2]},...
        mfilename, varName, fcnInputVarNumber);
else
    points = points.Location;
end

points = double(points);
end

%==========================================================================
function tf = checkMethod(method)

validatestring(method,{'falsecolor','blend','montage'},mfilename,'Method');

tf = true;
end

%==========================================================================
function tf = checkPlotOptions(options)

validateattributes(options,{'cell'}, {'size', [1 3]},...
    mfilename, 'PlotOptions');

validateattributes(options{1},{'char'},{},mfilename,'MarkerStyle1');
validateattributes(options{2},{'char'},{},mfilename,'MarkerStyle2');
validateattributes(options{3},{'char'},{},mfilename,'LineStyle');

% Now check valid strings
checkMarkerStyle(options{1}, 1);
checkMarkerStyle(options{2}, 2);

checkLineStyle(options{3});

tf = true;
end
%==========================================================================
function style=eliminateColorSpec(style)

colorSpec = cell2mat({'r','g','b','c','m','y','k','w'});

% Color can be specified only at the beginning or end of the style string.
% Look for only one specifier. If color was specified twice, it will cause
% a failure in later stages of parsing
if ~isempty(style)
    if isempty(strfind(colorSpec, style(1)))
        % try the other end
        if ~isempty(strfind(colorSpec, style(end)))
            style(end) = [];
        end
    else
        style(1) = [];
    end
end
end
%==========================================================================
function checkMarkerStyle(style, id)

style = strtrim(style); % remove blanks from either end of the string
style = strtrim(eliminateColorSpec(style)); % pull out valid color spec

if isempty(style)
   % permit empty marker style, which amounts to marker not being displayed 
else
    markerSpec = {'+','o','*','.','x','square','s','diamond','d','^',...
        'v','>','<','pentagram','p','hexagram','hImage'};
    
    try
        validatestring(style,markerSpec);
    catch %#ok<CTCH>
        error(message('vision:showMatchedFeatures:invalidMarkerStyle',id));
    end
end
end
%==========================================================================
function checkLineStyle(style)

style = strtrim(style); % remove blanks from either end of the string
style = strtrim(eliminateColorSpec(style)); % pull out valid color spec

if isempty(style)
    % permit empty line style thus letting plot use its default settings
else
    lineSpec = {'-','--',':','-.'};
    
    try
        validatestring(style,lineSpec);
    catch %#ok<CTCH>
        error(message('vision:showMatchedFeatures:invalidLineStyle'));
    end
end
end

