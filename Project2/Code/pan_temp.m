% location = "./"
% directory = dir(location)
% print(directory)

location = ".\Images\train_images\Set1\";
curr_img = imread(location + "1.jpg");
next_img = imread(location + "2.jpg");

    a=dir(location + '*.jpg');
    num_images=size(a,1);
%     tforms(num_images) = projective2d(eye(3));
    imageSize = zeros(num_images,2);
    prev_descriptors = 0;
    threshold = 200;

    
    % grayscale
     curr_img_grayscale = rgb2gray(curr_img);
     next_img_grayscale = rgb2gray(next_img);

     curr_img_corners = cornermetric(curr_img_grayscale);
     next_img_corners = cornermetric(next_img_grayscale);
     
     % imregional 
     curr_img_maxes = imregionalmax(curr_img_corners);
     next_img_maxes = imregionalmax(next_img_corners);

     % ANMS
     [curr_img_x_best, curr_img_y_best] = ANMS(curr_img, curr_img_maxes, curr_img_corners)
     [next_img_x_best, next_img_y_best] = ANMS(next_img, next_img_maxes, next_img_corners)

     % Display 
%       displayANMS(curr_img, curr_img_x_best, curr_img_y_best);
%       displayANMS(next_img, next_img_x_best, next_img_y_best);
     % descriptor
    curr_img_descriptors = feature_descriptors(curr_img_grayscale, curr_img_x_best, curr_img_y_best)
    next_img_descriptors = feature_descriptors(next_img_grayscale, next_img_x_best, next_img_y_best)
    
    % matching points
    threshold = 5;
    [matchedDescriptors1, matchedDescriptors2, matchedp1X, matchedp1Y, matchedp2X, matchedp2Y] = getMatchedPoints(curr_img_descriptors, next_img_descriptors, curr_img_x_best, curr_img_y_best, next_img_x_best, next_img_y_best, threshold)
    % matched lines 

    % RANSAC
    N_max = 200
    RANSAC_thresh = 10
    [H_ls, INLIERSp1X, INLIERSp1Y, INLIERSp2X, INLIERSp2Y] = RANSAC(N_max, RANSAC_thresh, matchedp1X, matchedp1Y, matchedp2X, matchedp2Y);



function [x_best, y_best] = ANMS(img, features, SHOW_OUTPUT)
    n_best = 300; 
    NStrong = 0;
    [y_size, x_size, ~] = size(img);
%    sz = size(img) 
    x_max = [];
    y_max = [];


    for x = 1:x_size
        for y = 1:y_size
            if features(y,x) == 1
                NStrong = NStrong + 1;
                x_max = [x_max x];
                y_max = [y_max y];  % array concatenation
            end
        end
    end
    
%     if SHOW_OUTPUT
%         % Plot features
%         imshow(img)
%         hold on
%         plot(features)
%         hold off
%     end
    
    r = Inf(1, NStrong);
    for i = 1:NStrong
        for j = 1:NStrong
            if SHOW_OUTPUT(y_max(j), x_max(j)) > SHOW_OUTPUT(y_max(i), x_max(i))
                ED = ((x_max(j) - x_max(i))^2) + ((y_max(j) - y_max(i))^2);
                if ED < r(i)
                    r(i) = ED;
                end
            end
        end
    end
    
    x_best = zeros(1, n_best);
    y_best = zeros(1, n_best);
    
    % lowest n_best points
    [~, I] = sort(r, 'descend');
    
%         x = x(idx(1:NBest));
%     y = y(idx(1:NBest));
    
    p = [x(:), y(:)];
    
    for i = 1:n_best
        indx = I(i);
        x_best(i) = x_max(indx);
        y_best(i) = y_max(indx);
    end
end

function displayANMS(img, x_best, y_best)
    figure
    imshow(img)
    hold on
%     plot(x_best, y_best, 'Color', 'r', 'Marker','.', 'LineStyle','-')
    plot(x_best, y_best, 'r.')
    hold off
end


function [descriptors] = feature_descriptors(img_grayscale, x_best, y_best)
    [~, n_best] = size(x_best);
    [y_size, x_size] = size(img_grayscale);
    descriptors = [];
    for i = 1:n_best
        % Patch is of size 41x41, so point is the actual center
        x = x_best(i);
        y = y_best(i);
        % TODO?
        if or(or(or(x <= 21, y<=21), x + 21> x_size), y + 21 > y_size)
            v = zeros(8*8,1);
            descriptors = [descriptors v];
            continue;
        end
        patch = img_grayscale(y-21:y+21, x-21:x+21);
        % Gaussian blur
        sig = 10;
        blurred = imgaussfilt(patch, sig);
        % Resize
        resized = imresize(blurred, [8 8]);
        % Reshape
        reshaped = double(reshape(resized, [64, 1]));
        % Now we need to standardize
        std_dev = std(reshaped);
        mean_reshaped = mean(reshaped);
        standardized = reshaped - mean_reshaped;
        standardized = standardized / std_dev;
        descriptors = [descriptors standardized];
    end
end


% fearture matching, first we need to make sure which points are matched
function [matchedDescriptors1, matchedDescriptors2, matchedp1X, matchedp1Y, matchedp2X, matchedp2Y] = getMatchedPoints(d1, d2, p1X, p1Y, p2X, p2Y, thresh)
    matchedp1X = [];
    matchedp1Y = [];
    matchedp2X = [];
    matchedp2Y = [];
    d1size = length(d1(1,:))
    d2size = length(d2(2,:))
    matchedDescriptors1 = [];
    matchedDescriptors2 = [];
    for i = 1:d1size
        for j = 1:d2size
            sumSquare(j) = sum((d1(:,i) - d2(:,j)).^2);
        end
        [sortedDist, I] = sort(sumSquare);
        ratio = sortedDist(1)/sortedDist(2);
        if (ratio < thresh)
            matchedDescriptors1 = [matchedDescriptors1, d1(:,i)];
            matchedDescriptors2 = [matchedDescriptors2, d2(:,I(1))];

            matchedp1X = [matchedp1X; p1X(i)];
            matchedp1Y = [matchedp1Y; p1Y(i)];

            matchedp2X = [matchedp2X; p2X(I(1))];
            matchedp2Y = [matchedp2Y; p2Y(I(1))];
        end
    end
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

for i = 1:length(x(:)),
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

function [H_ls, INLIERSp1X, INLIERSp1Y, INLIERSp2X, INLIERSp2Y] = RANSAC(N_max, thresh, matchedp1X, matchedp1Y, matchedp2X, matchedp2Y)
    total = size(matchedp1X, 1);
    INLIERSp1X = []; 
    INLIERSp1Y = []; 
    INLIERSp2X = []; 
    INLIERSp2Y = [];
    inliers_count = 0;
    iter = 0;

    while (iter < N_max || (inliers_count/total) < 0.90)
        random_i = randi([1 total], 1, 4)
        % 1. Select four pairs of matched pixels (at random), 1<=i<=4, p_1i/p_2i from images 1/2.
        
        tempX1 = matchedp1X(random_i)';
        tempY1 = matchedp1Y(random_i)';
        tempX2 = matchedp2X(random_i)';
        tempY2 = matchedp2Y(random_i)';
        % 2. Compute the homography matrix from the four feature pairs using est_homography
        H = est_homography(tempX1,tempY1,tempX2,tempY2)
        
        % 3. Compute inliers where SSD(Hp_1, p_2) < threshold
        
        fprintf("");
        
        for i=1:4
            p1 = [tempX1(i); tempY1(i); 1]
            p2 = [tempX2(i); tempY2(i); 1]
            Hp1 = H*p1
            X = Hp1 - p2
            ssd = sum(X(:).^2)
            if (ssd < thresh)
                INLIERSp1X = [INLIERSp1X; tempX1(i)];
                INLIERSp1Y = [INLIERSp1Y; tempY1(i)];
                INLIERSp2X = [INLIERSp2X; tempX2(i)];
                INLIERSp2Y = [INLIERSp2Y; tempY2(i)];
                inliers_count = inliers_count + 1
            end
        end
        % 4. Repeat the last 3 steps until you reach N_max iters or 90% of inliers.
        iter = iter + 1;
        % 5. Keep largest set of inliers.
    end

    % https://www.mathworks.com/matlabcentral/answers/373747-find-unique-in-matrix-of-x-y-coordinates
    INLIERSp1XY = [INLIERSp1X, INLIERSp1Y]
    INLIERSp2XY = [INLIERSp2X, INLIERSp2Y]
    uniqueINLIERSp1XY = unique(INLIERSp1XY, 'rows', 'stable')
    uniqueINLIERSp2XY = unique(INLIERSp2XY, 'rows', 'stable')

    % 6. Recompute least-square H on all inliers.
    H_ls = est_homography(uniqueINLIERSp1XY(:,1),uniqueINLIERSp1XY(:,2),uniqueINLIERSp2XY(:,1),uniqueINLIERSp2XY(:,2))

    fprintf("");
end
