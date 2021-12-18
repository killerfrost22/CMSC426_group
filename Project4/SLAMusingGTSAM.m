 function [LandMarksComputed, AllPosesComputed] = SLAMusingGTSAM(DetAll, K, TagSize, TLeftImgs);
	% For Input and Output specifications refer to the project pdf
    
	import gtsam.*
    % https://gtsam.org/tutorials/intro.html
    
	% Refer to Factor Graphs and GTSAM Introduction
	% https://research.cc.gatech.edu/borg/sites/edu.borg/files/downloads/gtsam.pdf
	% and the examples in the library in the GTSAM toolkit. See folder
	% gtsam_toolbox/gtsam_examples
    LandMarksComputed = [];
    
    %% Homography matrix
    intrinsics = cameraParameters('IntrinsicMatrix',K');
    %Setting up initial
        %First frame
    frame1 = DetAll{1};
    frame1 = sortrows(frame1,1);
        %First tag
    initial = frame1(1,:);

    World = [[0,0];[TagSize,0];[TagSize,TagSize];[0,TagSize]];
    IMG = [[initial(2),initial(3)];[initial(4),initial(5)];[initial(6),initial(7)];[initial(8),initial(9)]];
    
    % Get Homography matrix
    H = est_homography(IMG(:,1),IMG(:,2),World(:,1),World(:,2));
    
    Hp = inv(K) * H;

    Rot = [Hp(:,1), Hp(:,2), cross(Hp(:,1),Hp(:,2))];
    
    [U, ~, V] = svd(Rot);
    R = U*[1,0,0;0,1,0;0,0,det(U*V')]*V';
    T = Hp(:,3)/(norm(Hp(:,1)));
    x0 = -R'*T;
    AllPosesComputed(1,:) = x0';
    
    Data.R{1} = R;
    Data.T{1} = T;
    
    % landmarks
    for k=1:size(DetAll{1})
       tag = frame1(k,:);
       IMG = [[tag(2),tag(3)];[tag(4),tag(5)];[tag(6),tag(7)];[tag(8),tag(9)]];
       LandMarks = pointsToWorld(intrinsics,R',T,IMG);
       LandMarksComputed = [LandMarksComputed; [tag(1), reshape(LandMarks.',1,[])]];
    end
    
    %% Iterate through frames
    for frame=2:length(DetAll)
        lastX = []; lastY = []; currX = []; currY = [];
        
        lastFrame = LandMarksComputed;
        currFrame = sortrows(DetAll{frame}, 1);
        for currTag=1:size(currFrame,1)
            for lastTag=1:size(lastFrame,1)
                if (lastFrame(lastTag, 1) == currFrame(currTag, 1))
                    lastX = [lastX; lastFrame(lastTag, 2); lastFrame(lastTag, 4); lastFrame(lastTag, 6); lastFrame(lastTag, 8)];
                    lastY = [lastY; lastFrame(lastTag, 3); lastFrame(lastTag, 5); lastFrame(lastTag, 7); lastFrame(lastTag, 9)];
                    currX = [currX; currFrame(currTag, 2); currFrame(currTag, 4); currFrame(currTag, 6); currFrame(currTag, 8)];
                    currY = [currY; currFrame(currTag, 3); currFrame(currTag, 5); currFrame(currTag, 7); currFrame(currTag, 9)];  
                end
            end
        end
 
        
        H = est_homography(currX, currY, lastX, lastY);
        Hp = inv(K) * H;

        Rot = [Hp(:,1), Hp(:,2), cross(Hp(:,1),Hp(:,2))];
        
        [U, ~, V] = svd(Rot);
        R = U*[1,0,0;0,1,0;0,0,det(U*V')]*V';
        T = Hp(:,3)/(norm(Hp(:,1)));
        x = -R'*T;
        AllPosesComputed(frame,:) = x';
        Data.R{frame} = R;
        Data.T{frame} = T;
        
        % landmarks for each tag
        for k=1:size(DetAll{frame})
           tag = currFrame(k,:);
           if ~ismember(tag(1),LandMarksComputed(:,1))
               IMG = [[tag(2),tag(3)];[tag(4),tag(5)];[tag(6),tag(7)];[tag(8),tag(9)]];
               LandMarks = pointsToWorld(intrinsics,R',T,IMG);
               LandMarksComputed = [LandMarksComputed; [tag(1), reshape(LandMarks.',1,[])]];
           end
        end
    end
    LandMarksComputed = sortrows(LandMarksComputed,1);
    % Since we know Z is always positive, we avoid sign errors by taking
    % the absolute value of the Z coordinate
    AllPosesComputed(:,3) = abs(AllPosesComputed(:,3));
    
    %% Plot
    plotPoints = true;
    if plotPoints
        plot3(AllPosesComputed(:,1),AllPosesComputed(:,2),AllPosesComputed(:,3),'o');
        hold on;
        plot3(LandMarksComputed(:,2),LandMarksComputed(:,3), zeros(81,1), 'r*');
        plot3(LandMarksComputed(:,4),LandMarksComputed(:,5), zeros(81,1),'b*');
        plot3(LandMarksComputed(:,6),LandMarksComputed(:,7), zeros(81,1),'green*');
        plot3(LandMarksComputed(:,8),LandMarksComputed(:,9), zeros(81,1),'black*');
        hold off;
    end
    
    %% GTSAM 
    graph = NonlinearFactorGraph;
    initialEst = Values;
    
    pointPriorNoise = noiseModel.Diagonal.Sigmas(ones(3,1) * 1e-6);
    pointNoise = noiseModel.Diagonal.Sigmas(ones(3,1) * 1e-6);
    posePriorNoise = noiseModel.Diagonal.Sigmas(ones(6,1) * .001);
    poseNoise = noiseModel.Diagonal.Sigmas(ones(6,1) * .001);
    measurementNoise = noiseModel.Isotropic.Sigma(2,1.0);
    
    % Add prior for first pose
    graph.add(PriorFactorPose3(symbol('x',1),...
        Pose3(Rot3(Data.R{1}), Point3(Data.T{1})),...
        posePriorNoise));
    
   
    % Add prior for world origin
    graph.add(PriorFactorPoint3(symbol('L',10),...
        Point3(0,0,0),...
        pointPriorNoise));

    % Add identity between factor between poses
    for i = 1:size(DetAll,2)-1
        p1 = Pose3(Rot3(Data.R{i}),Point3(Data.T{i}));
        p2 = Pose3(Rot3(Data.R{i+1}),Point3(Data.T{i+1}));
        bt = p1.between(p2);
        
        id = Pose3(eye(4,4));
        
        graph.add(BetweenFactorPose3(symbol('x',i),symbol('x',i+1),...
            id,...
            poseNoise));
    end
    
    % projection factors
    Kp = Cal3_S2(K(1, 1), K(2, 2), 0, K(1,3), K(2, 3));
    for i = 1:size(DetAll,2)
        frame = DetAll{i};
        for j = 1:size(frame,1)
            row = frame(j,:);
            graph.add(GenericProjectionFactorCal3_S2(...
                Point2(row(2),row(3)), measurementNoise, symbol('x',i),...
                symbol('L',row(1)), Kp));
            graph.add(GenericProjectionFactorCal3_S2(...
                Point2(row(4),row(5)), measurementNoise, symbol('x',i),...
                symbol('M',row(1)), Kp));
            graph.add(GenericProjectionFactorCal3_S2(...
                Point2(row(6),row(7)), measurementNoise, symbol('x',i),...
                symbol('N',row(1)), Kp));
            graph.add(GenericProjectionFactorCal3_S2(...
                Point2(row(8),row(9)), measurementNoise, symbol('x',i),...
                symbol('O',row(1)), Kp));
        end
    end
    
    % Add between factors between tags
    for i = 1:size(LandMarksComputed,1)
        graph.add(BetweenFactorPoint3(symbol('L',LandMarksComputed(i,1)),...
            symbol('M',LandMarksComputed(i,1)),Point3(TagSize, 0, 0),pointPriorNoise));
        %graph.add(BetweenFactorPoint3(symbol('L',LandMarksComputed(i,1)),...
            %symbol('N',LandMarksComputed(i,1)),Point3(sqrt(2)*TagSize, sqrt(2)*TagSize, 0),pointPriorNoise));
        graph.add(BetweenFactorPoint3(symbol('L',LandMarksComputed(i,1)),...
            symbol('O',LandMarksComputed(i,1)),Point3(0, TagSize, 0),pointPriorNoise));
        graph.add(BetweenFactorPoint3(symbol('M',LandMarksComputed(i,1)),...
            symbol('N',LandMarksComputed(i,1)),Point3(TagSize, 0, 0),pointPriorNoise));
        %graph.add(BetweenFactorPoint3(symbol('O',LandMarksComputed(i,1)),...
            %symbol('M',LandMarksComputed(i,1)),Point3(sqrt(2)*TagSize, sqrt(2)*TagSize, 0),pointPriorNoise));
        graph.add(BetweenFactorPoint3(symbol('O',LandMarksComputed(i,1)),...
            symbol('N',LandMarksComputed(i,1)),Point3(0, TagSize, 0),pointPriorNoise));
    end
    
    %graph.print(sprintf('\nFactor Graph: '));
    
    for i = 1:size(DetAll,2)
        initialEst.insert(symbol('x',i), Pose3(Rot3(Data.R{i}), Point3(Data.T{i})));
    end
    
    for i = 1:size(LandMarksComputed,1)
       initialEst.insert(symbol('L',LandMarksComputed(i, 1)),Point3([LandMarksComputed(i, 2:3) 0]'));
       initialEst.insert(symbol('M',LandMarksComputed(i, 1)),Point3([LandMarksComputed(i, 4:5) 0]'));
       initialEst.insert(symbol('N',LandMarksComputed(i, 1)),Point3([LandMarksComputed(i, 6:7) 0]'));
       initialEst.insert(symbol('O',LandMarksComputed(i, 1)),Point3([LandMarksComputed(i, 8:9) 0]')); 
    end
    
    parameters = LevenbergMarquardtParams;
    parameters.setlambdaInitial(1.0);
    parameters.setVerbosityLM('trylambda');
    optimizer = LevenbergMarquardtOptimizer(graph, initialEst, parameters);
    result = optimizer.optimize();
    
    
    %% Retrieving landmarks
    for i = 1:size(LandMarksComputed,1)
        pL(i, :) = [result.at(symbol('L', LandMarksComputed(i, 1))).x result.at(symbol('L', LandMarksComputed(i, 1))).y result.at(symbol('L', LandMarksComputed(i, 1))).z];
        pM(i, :) = [result.at(symbol('M', LandMarksComputed(i, 1))).x result.at(symbol('M', LandMarksComputed(i, 1))).y result.at(symbol('M', LandMarksComputed(i, 1))).z];
        pN(i, :) = [result.at(symbol('N', LandMarksComputed(i, 1))).x result.at(symbol('N', LandMarksComputed(i, 1))).y result.at(symbol('N', LandMarksComputed(i, 1))).z];
        pO(i, :) = [result.at(symbol('O', LandMarksComputed(i, 1))).x result.at(symbol('O', LandMarksComputed(i, 1))).y result.at(symbol('O', LandMarksComputed(i, 1))).z];
    end
    LandMarksComputed = [LandMarksComputed(:, 1) pL pM pN pO];
    
    % Retrieving poses
    AllPosesComputed = [];
    for frame = 1:size(DetAll,2)
        pose = result.at(symbol('x', frame));
        r = pose.rotation.matrix;
        t = pose.translation.vector;
        x = -r'*t;
        AllPosesComputed(frame,:) = x';
    end
    AllPosesComputed(:,3) = abs(AllPosesComputed(:,3));
    
    %% Plot
    figure(2);
    if plotPoints
        plot3(AllPosesComputed(:,1),AllPosesComputed(:,2),AllPosesComputed(:,3),'o');
        hold on;
        plot3(LandMarksComputed(:,2),LandMarksComputed(:,3), LandMarksComputed(:,4), 'r*');
        plot3(LandMarksComputed(:,5),LandMarksComputed(:,6), LandMarksComputed(:,7),'b*');
        plot3(LandMarksComputed(:,8),LandMarksComputed(:,9), LandMarksComputed(:,10),'green*');
        plot3(LandMarksComputed(:,11),LandMarksComputed(:,12), LandMarksComputed(:,13),'black*');
        
        
        hold off;
    end
end