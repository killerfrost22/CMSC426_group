function [pano] = MyPanorama()

%% YOUR CODE HERE.
% Must load images from ../Images/Input/
% Must return the finished panorama.
% sprintf('Hi there!')

    location = "..\Images\train_images\Set1\";
    curr_img = imread(location + "1.jpg"); 
    next_img = imread(location + "2.jpg");

    a=dir(location + '*.jpg');
    num_images=size(a,1)
%     tforms(num_images) = projective2d(eye(3));
    imageSize = zeros(num_images,2);
    prev_descriptors = 0;


    for i =1:num_images
     % grayscacurr_img_grayscalele
      = rgb2gray(curr_img);
     next_img_grayscale = rgb2gray(next_img);

     curr_img_corners = cornermetric(curr_img_grayscale);
     next_img_corners = cornermetric(next_img_grayscale);
     
     % imregional 
     curr_img_maxes = imregionalmax(curr_img_corners);
     next_img_maxes = imregionalmax(next_img_corners);

     % ANMS
     [curr_img_x_best, curr_img_y_best] = ANMS(curr_img, curr_img_maxes, curr_img_corners);
     [next_img_x_best, next_img_y_best] = ANMS(next_img, next_img_maxes, next_img_corners);

     % Display 
     displayANMS(curr_img, curr_img_x_best, curr_img_y_best)
     displayANMS(next_img, next_img_x_best, next_img_y_best)
end


function [x_best, y_best] = ANMS(img, features, SHOW_OUTPUT)
    n_best = 300; 
    NStrong = 0;
    [y_size, x_size, ~] = size(img)
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


