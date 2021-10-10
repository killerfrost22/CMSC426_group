
function [pano] = MyPanorama()

%% YOUR CODE HERE.
% Must load images from ../Images/Input/
% Must return the finished panorama.
%     location = "..\Images\train_images\Set1\";
%     curr_img = imread(location + "1.jpg");
%     next_img = imread(location + "2.jpg");
% 
%     all=dir(location + '*.jpg');
%     number_images = size(all);
%     
%     curr_img_grayscale = rgb2gray(curr_img);
%     next_img_grayscale = rgb2gray(next_img);
    
    
    for img = 2:imgN
        % ANMS    
        I1 = getImage(img-1, path);
        I2 = getImage(img, path);
        imageSize(img,:) = size(I2);
        
        p1 = ANMS(rgb2gray(I1), N_Best, SHOW_OUTPUT);
        p2 = ANMS(rgb2gray(I2), N_Best, SHOW_OUTPUT);
       
        p1
    end
    
%         %% Feature Descriptor
%         % Get filter
%         H = fspecial(FILTER);
%         
%         D1 = getFeatureDescriptors(p1, H, I1);
%         D2 = getFeatureDescriptors(p2, H, I2);
end

function img = getImage(i, path)
    img = imread(fullfile(path(i).folder, path(i).name));
end


function p = ANMS(I, NBest, SHOW_OUTPUT) 
    cornerScoreImg = cornermetric(I); 
    features = imregionalmax(cornerScoreImg);
    sz = size(I); 
    x = [];
    y = [];
    for i = 20:sz(1)-20
        for j = 20:sz(2)-20
            if(features(i,j) == 1)
                x = [x; j];
                y = [y; i];
            end
        end
    end   
    for i = 1:NStrong
     for j = 1:NStrong
        % We check if the metric scores are bigger between the current
        % point and the previous, if it is we then get the distances. We
        % then iteratively get smaller distances.
        if cornerScoreImg(y(j), x(j)) > cornerScoreImg(y(i), x(i))
          % Calculate distance
          ED = (y(j) - y(i))^2 + (x(j) - x(i))^2;
          if (ED < radius(i))
            radius(i) = ED;
          end
        end
     end
    end
    %sort the radius descending order
    [~,idx] = sort(radius, 'descend');
    %The amount will be assigned the 
    if SHOW_OUTPUT
        imshow(I)
        hold on
        plot(features)
        hold off
    end
end

 function [] = displayANMS(img, x_best, y_best)
     imshow(img);
     hold on
     
     plot(x_best, y_best, 'r.');
     hold off
 end


function [descriptors] = feature_descriptors(img_gray, x_best, y_best)
    [~, n_best] = size(x_best);
    [y_size, x_size] = size(img_gray);
    descriptors = [];
    for num = 1:n_best
        % Patch is of size 40x40, so point is the actual center therfore
        % 41*41
        x = x_best(num);
        y = y_best(inumndx);
        % TODO handle patches near the edge?
        if or(or(or(x <= 20, y<=20), x + 20 > x_size), y + 20 > y_size)
            v = zeros(64,1);
            descriptors = [descriptors v];
            continue;
        end
        patch = img_gray(y-20:y+20, x-20:x+20);
        % Now, we take the gaussian blur!
        sig = 10;
        blurred = imgaussfilt(patch, sig);
        % size
        resized = imresize(blurred, [8 8]);
        % shape
        reshaped = double(reshape(resized, [64, 1]));
        % standardize
        std_dev = std(reshaped);
        mean_reshaped = mean(reshaped);
        standardized = reshaped - mean_reshaped;
        standardized = standardized / std_dev;
        descriptors = [descriptors standardized];
    end
end

function [pt1, pt2, best_matches] = feature_matching(img1_descriptors, img2_descriptors, img1_x_best, img1_y_best, img2_x_best, img2_y_best)
    best_matches = [];
    
    %SSD is the sum of squared difference 
    [~, n_best] = size(img1_x_best);
    for i = 1:n_best  %the indx of the img1 descriptors
        best_j = -1;
        best_ssd = -1;
        second_best_j = -1;
        second_best_ssd = -1;
        if img1_descriptors(64, i) == 0 && img1_descriptors(37, i) == 0 && img1_descriptors(1, i) == 0
            continue
        end
        desc1 = img1_descriptors(:, i);
        for j = 1:n_best  % img2 descriptors
            if img2_descriptors(64, j) == 0 && img2_descriptors(37, j) == 0 && img2_descriptors(1, j) == 0
                % If descriptor is all zeros, skip it
                continue
            end
            desc2 = img2_descriptors(:, j);
            diff = double(desc1) - double(desc2);
            ssd = sum(diff(:).^2);
            if or(best_j == -1, ssd < best_ssd)
                second_best_ssd = best_ssd;
                second_best_j = best_j;
                best_ssd = ssd;
                best_j = j;
            elseif or(second_best_j == -1, ssd < second_best_ssd)
                second_best_ssd = ssd;
                second_best_j = j;
            end
        end
       
        if second_best_ssd - best_ssd > 0.5
            v = reshape([i best_j (second_best_ssd - best_ssd)], [3 1]);
            best_matches = [best_matches v];
        end
    end
    
    [~, num_best_matches] = size(best_matches);
    pt1 = zeros(num_best_matches, 2);
    pt2 = zeros(num_best_matches, 2);
    best_matches = sortrows(best_matches.',3).';
    % Format
    for indx = 1:num_best_matches
        match = best_matches(:, indx);
        img1_x = img1_x_best(match(1));
        img1_y = img1_y_best(match(1));
        img2_x = img2_x_best(match(2));
        img2_y = img2_y_best(match(2));
        % vectors to be vectorization 
        pt1(indx, 1) = img1_x;
        pt1(indx, 2) = img1_y;
        pt2(indx, 1) = img2_x;
        pt2(indx, 2) = img2_y;
    end
end
