function [descriptors, X, Y] = feature_descriptors(img_grayscale, x, y)
    [~, n_best] = size(x);
    [ x_size, y_size] = size(img_grayscale);
    descriptors = []; X = []; Y = [];
    for i = 1:n_best
        % Patch is of size 41x41, so point is the actual center
        x_i = x(i);
        y_i = y(i);

        if (((x_i-19) >= 1) &&  ((y_i-19)>=1) && ((x_i + 20) <= x_size) && ((y_i + 20) <= y_size))
            
            patch = img_grayscale(x_i-19:x_i+20, y_i-19:y_i+20);
            
%             imshow(patch)
%             figure(10,10); imshow(blurred)

%             avg5 = ones(5)/9;
%             C = conv2(avg5,blurred )
            
%             H = fspecial('motion', 10, 45);
                
            % h = fspecial('gaussian',hsize,sigma) returns 
            % a rotationally symmetric Gaussian lowpass filter of size hsize 
            % with standard deviation sigma. Not recommended.
            
            hsize = 10; sigma = 2;
%             H = fspecial('gaussian',hsize,sigma);
%             blurred = imfilter(patch,H,'replicate'); 
%             imshow(blurred);

            blurred = imgaussfilt(patch, sigma);


%             resized = imresize(blurred, [8 8]);
            resized = blurred(1:5:end, 1:5:end);
            
            reshaped = double(reshape(resized, [64, 1]));
            
            std_dev = std(reshaped);
            mean_reshaped = mean(reshaped);
            standardized = (reshaped - mean_reshaped) ./ std_dev;
            descriptors = [descriptors standardized];
            X = [X y_i];
            Y = [Y x_i];
            disp("")
        end
    end
    disp("")
end
