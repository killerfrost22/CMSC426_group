function dist = measureDepth(cluster)
    % training section
    path = dir('train_images/masks/*.bmp');
    numFiles = length(path);
    distance = zeros(numFiles, 1);
    area = zeros(numFiles, 1);
    for i = 1:numFiles
        % Getting the file name
        currImgPath = fullfile(path(i).folder, path(i).name);
        currImg = imread(currImgPath);
        [rows, cols, ~] = size(cluster);   
        name = strsplit(path(i).name, '-mask.');   
        name = string(name{1});
        
        % Distance of ball in centimeters
        distance(i) = str2double(name);
    
        % Getting the area of the ball from mask
        n = 0;
        for row=1:rows
            for col=1:cols
                if (currImg(row,col) == 255)
                    n = n + 1;
                end
            end
        end
        area(i) = n;
    end
    %fittedCurve = fit(area,distance,'poly2'); 
    fittedCurve = polyfit(area,distance,2); 
    %Find the area of the circle
    %testImg = imread(cluster);
    testImg = cluster;
    [rows, cols, ~] = size(testImg);
    
    n = 0;
    for row = 1:rows
        for col = 1:cols
            if(testImg(row,col) ~= 0)
                n = n + 1;
            end
        end
    end
    dist = polyval(fittedCurve,n);
end