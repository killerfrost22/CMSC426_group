% Yizhan & Yingqiao
function [LandMarksComputed, AllPosesComputed] = SLAMusingGTSAM(DetAll, K, TagSize, TLeftImgs);
	% For Input and Output specifications refer to the project pdf
    
	import gtsam.*
    % https://gtsam.org/tutorials/intro.html
    
	% Refer to Factor Graphs and GTSAM Introduction
	% https://research.cc.gatech.edu/borg/sites/edu.borg/files/downloads/gtsam.pdf
	% and the examples in the library in the GTSAM toolkit. See folder
	% gtsam_toolbox/gtsam_examples
    LandMarksComputed = [];
    AllPosesComputed = [];

    intrinsics = cameraParameters('IntrinsicMatrix',K');
    % Initialize the first frame and tag
    fst_frame = DetAll{1};
    fst_frame = sortrows(fst_frame,1);
    fst = fst_frame(1,:);

    % World Coordinates
    Co_Wd = [[0,0];[TagSize,0];[TagSize,TagSize];[0,TagSize]];
    % Image Coordinates
    Co_Img = [[fst(2),fst(3)];[fst(4),fst(5)];[fst(6),fst(7)];[fst(8),fst(9)]];
    
    % Detection data stored as [TagID, p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y]
    first_col = fst_frame(:, 1);
    tag_10_data = fst_frame(first_col == 10, :);
    p1x = tag_10_data(2);
    p1y = tag_10_data(3);
    p2x = tag_10_data(4);
    p2y = tag_10_data(5);
    p3x = tag_10_data(6);
    p3y = tag_10_data(7);
    p4x = tag_10_data(8);
    p4y = tag_10_data(9);
    tag_10_coords = [p1x p1y; p2x p2y; p3x p3y; p4x p4y];
    origin_coords = [0 0; TagSize 0; TagSize TagSize; 0 TagSize];
    
    % Calculate the homography matrix
    tform = estimateGeometricTransform(tag_10_coords, origin_coords, 'projective');

    H = est_homography(Co_Img(:,1),Co_Img(:,2),Co_Wd(:,1),Co_Wd(:,2));
    Hp = inv(K) * H;
    
    [U, ~, V] = svd([Hp(:,1), Hp(:,2), cross(Hp(:,1),Hp(:,2))]);
    R = U*[1,0,0;0,1,0;0,0,det(U*V')]*V';
    T = Hp(:,3)/(norm(Hp(:,1)));
    AllPosesComputed(1,:) = (-R'*T)';
    
    Data.R{1} = R;
    Data.T{1} = T;
    
    % Record the landmarks
    for k=1:size(DetAll{1})
       tag = fst_frame(k,:);
       Co_Img = [[tag(2),tag(3)];[tag(4),tag(5)];[tag(6),tag(7)];[tag(8),tag(9)]];
       LandMarks = pointsToWorld(intrinsics,R',T,Co_Img);
       LandMarksComputed = [LandMarksComputed; [tag(1), reshape(LandMarks.',1,[])]];
    end
    

    % Iterate through frames
    for frame=2:length(DetAll)
        pre_X = []; pre_Y = []; curr_X = []; curr_Y = [];
        pre_Fr = LandMarksComputed;
        curr_Fr = sortrows(DetAll{frame}, 1);
        for curr_Tag=1:size(curr_Fr,1)
            for lastTag=1:size(pre_Fr,1)
                if (pre_Fr(lastTag, 1) == curr_Fr(curr_Tag, 1))
                    pre_X = [pre_X; pre_Fr(lastTag, 2); pre_Fr(lastTag, 4); pre_Fr(lastTag, 6); pre_Fr(lastTag, 8)];
                    pre_Y = [pre_Y; pre_Fr(lastTag, 3); pre_Fr(lastTag, 5); pre_Fr(lastTag, 7); pre_Fr(lastTag, 9)];
                    curr_X = [curr_X; curr_Fr(curr_Tag, 2); curr_Fr(curr_Tag, 4); curr_Fr(curr_Tag, 6); curr_Fr(curr_Tag, 8)];
                    curr_Y = [curr_Y; curr_Fr(curr_Tag, 3); curr_Fr(curr_Tag, 5); curr_Fr(curr_Tag, 7); curr_Fr(curr_Tag, 9)];  
                end
            end
        end
 
        
        H = est_homography(curr_X, curr_Y, pre_X, pre_Y);
        Hp = inv(K) * H;

        Rot = [Hp(:,1), Hp(:,2), cross(Hp(:,1),Hp(:,2))];
        
        [U, ~, V] = svd(Rot);
        R = U*[1,0,0;0,1,0;0,0,det(U*V')]*V';
        T = Hp(:,3)/(norm(Hp(:,1)));
        x = -R'*T;
        AllPosesComputed(frame,:) = x';
        Data.R{frame} = R;
        Data.T{frame} = T;
        
        % get the landmark for each tag
        for k=1:size(DetAll{frame})
           tag = curr_Fr(k,:);
           if ~ismember(tag(1),LandMarksComputed(:,1))
               Co_Img = [[tag(2),tag(3)];[tag(4),tag(5)];[tag(6),tag(7)];[tag(8),tag(9)]];
               LandMarks = pointsToWorld(intrinsics,R',T,Co_Img);
               LandMarksComputed = [LandMarksComputed; [tag(1), reshape(LandMarks.',1,[])]];
           end
        end
    end
    LandMarksComputed = sortrows(LandMarksComputed,1);
    % Z is always positive in our case, so we take the abs value in function
    AllPosesComputed(:,3) = abs(AllPosesComputed(:,3));
    
    %% Plot the first two figures (PRE-GTSAM)
    figure(1);

    plotPoints = true;
    if plotPoints
        plot3(AllPosesComputed(:,1),AllPosesComputed(:,2),AllPosesComputed(:,3),'o');
        hold on;
        title('Without-GTSAM-no pose');
        plot3(LandMarksComputed(:,2),LandMarksComputed(:,9), zeros(81,1), 'green*');
        hold off;
    end
    
    figure(2);
    if plotPoints
        plot3(AllPosesComputed(:,1),AllPosesComputed(:,2),AllPosesComputed(:,3),'o');
        hold on;
        title('Without-GTSAM-no pose with landmarks');
        plot3(LandMarksComputed(:,2),LandMarksComputed(:,3), zeros(81,1), 'r*');
        plot3(LandMarksComputed(:,4),LandMarksComputed(:,5), zeros(81,1),'b*');
        plot3(LandMarksComputed(:,6),LandMarksComputed(:,7), zeros(81,1),'green*');
        plot3(LandMarksComputed(:,8),LandMarksComputed(:,9), zeros(81,1),'black*');
        % Must not exceed tthe boundary of 9
        % plot3(LandMarksComputed(:,10),LandMarksComputed(:,11), zeros(81,1), 'purple*');
        legend('show')
        hold off;
    end
    
    %% GTSAM 
    graph = NonlinearFactorGraph;
    fstEst = Values;
    
    % Add prior for first pose
    graph.add(PriorFactorPose3(symbol('x',1),Pose3(Rot3(Data.R{1}), Point3(Data.T{1})), noiseModel.Diagonal.Sigmas(ones(6,1) * .001)));
    
   
    % Add prior for Co_Wd origin
    graph.add(PriorFactorPoint3(symbol('L',10),Point3(0,0,0),noiseModel.Diagonal.Sigmas(ones(3,1) * 1e-6)));

    % Add identity between factor between poses
    for i = 1:size(DetAll,2)-1
        p1 = Pose3(Rot3(Data.R{i}),Point3(Data.T{i}));
        p2 = Pose3(Rot3(Data.R{i+1}),Point3(Data.T{i+1}));
        bt = p1.between(p2);
        
        id = Pose3(eye(4,4));
        
        graph.add(BetweenFactorPose3(symbol('x',i),symbol('x',i+1),...
            id,...
            noiseModel.Diagonal.Sigmas(ones(6,1) * .001)));
    end
    
    % projection factors
    Kp = Cal3_S2(K(1, 1), K(2, 2), 0, K(1,3), K(2, 3));
    for i = 1:size(DetAll,2)
        frame = DetAll{i};
        for j = 1:size(frame,1)
            row = frame(j,:);
            graph.add(GenericProjectionFactorCal3_S2(...
                Point2(row(2),row(3)), noiseModel.Isotropic.Sigma(2,1.0), symbol('x',i),...
                symbol('L',row(1)), Kp));
            graph.add(GenericProjectionFactorCal3_S2(...
                Point2(row(4),row(5)), noiseModel.Isotropic.Sigma(2,1.0), symbol('x',i),...
                symbol('M',row(1)), Kp));
            graph.add(GenericProjectionFactorCal3_S2(...
                Point2(row(6),row(7)), noiseModel.Isotropic.Sigma(2,1.0), symbol('x',i),...
                symbol('N',row(1)), Kp));
            graph.add(GenericProjectionFactorCal3_S2(...
                Point2(row(8),row(9)), noiseModel.Isotropic.Sigma(2,1.0), symbol('x',i),...
                symbol('O',row(1)), Kp));
        end
    end
    
    % Add between factors between tags
    for i = 1:size(LandMarksComputed,1)
        graph.add(BetweenFactorPoint3(symbol('L',LandMarksComputed(i,1)),...
            symbol('M',LandMarksComputed(i,1)),Point3(TagSize, 0, 0),noiseModel.Diagonal.Sigmas(ones(3,1) * 1e-6)));
        graph.add(BetweenFactorPoint3(symbol('L',LandMarksComputed(i,1)),...
            symbol('O',LandMarksComputed(i,1)),Point3(0, TagSize, 0),noiseModel.Diagonal.Sigmas(ones(3,1) * 1e-6)));
        graph.add(BetweenFactorPoint3(symbol('M',LandMarksComputed(i,1)),...
            symbol('N',LandMarksComputed(i,1)),Point3(TagSize, 0, 0),noiseModel.Diagonal.Sigmas(ones(3,1) * 1e-6)));
        graph.add(BetweenFactorPoint3(symbol('O',LandMarksComputed(i,1)),...
            symbol('N',LandMarksComputed(i,1)),Point3(0, TagSize, 0),noiseModel.Diagonal.Sigmas(ones(3,1) * 1e-6)));
    end
    
    % graph.print(sprintf('\nFactor Graph: '));
    
    for i = 1:size(DetAll,2)
        fstEst.insert(symbol('x',i), Pose3(Rot3(Data.R{i}), Point3(Data.T{i})));
    end
    
    
%% Optimize using Dogleg (failed)
% params = DoglegParams;
% params.setAbsoluteErrorTol(1e-15);
% params.setRelativeErrorTol(1e-15);
% params.setVerbosity('ERROR');
% params.setVerbosityDL('VERBOSE');
% params.setOrdering(graph.orderingCOLAMD());
% optimizer = DoglegOptimizer(graph, fstEstimate, params);
% 
% result = optimizer.optimizeSafely();
% result.print('final result');
%%
    for i = 1:size(LandMarksComputed,1)
       fstEst.insert(symbol('L',LandMarksComputed(i, 1)),Point3([LandMarksComputed(i, 2:3) 0]'));
       fstEst.insert(symbol('M',LandMarksComputed(i, 1)),Point3([LandMarksComputed(i, 4:5) 0]'));
       fstEst.insert(symbol('N',LandMarksComputed(i, 1)),Point3([LandMarksComputed(i, 6:7) 0]'));
       fstEst.insert(symbol('O',LandMarksComputed(i, 1)),Point3([LandMarksComputed(i, 8:9) 0]')); 
    end
    
    parameters = LevenbergMarquardtParams;
    parameters.setlambdaInitial(1.0);
    parameters.setVerbosityLM('trylambda');
    optimizer = LevenbergMarquardtOptimizer(graph, fstEst, parameters);
    result = optimizer.optimize();
    
%     marginals = Marginals(graph, result);
%     cla
%     hold on;
%     plot3DPoints(result, []);  
%     plot3DTrajectory(result, '*', 1, 8);
%     hold off
    %% Retrieving landmarks
    for i = 1:size(LandMarksComputed,1)
        pL(i, :) = [result.at(symbol('L', LandMarksComputed(i, 1))).x result.at(symbol('L', LandMarksComputed(i, 1))).y result.at(symbol('L', LandMarksComputed(i, 1))).z];
        pM(i, :) = [result.at(symbol('M', LandMarksComputed(i, 1))).x result.at(symbol('M', LandMarksComputed(i, 1))).y result.at(symbol('M', LandMarksComputed(i, 1))).z];
        pN(i, :) = [result.at(symbol('N', LandMarksComputed(i, 1))).x result.at(symbol('N', LandMarksComputed(i, 1))).y result.at(symbol('N', LandMarksComputed(i, 1))).z];
        pO(i, :) = [result.at(symbol('O', LandMarksComputed(i, 1))).x result.at(symbol('O', LandMarksComputed(i, 1))).y result.at(symbol('O', LandMarksComputed(i, 1))).z];
    end
    LandMarksComputed = [LandMarksComputed(:, 1) pL pM pN pO];
    
    % Retrieving poses

%     for frame = 1:size(DetAll,2)
%         pose = result.at(symbol('x', frame));
%         r = pose.rotation.matrix;
%         t = pose.translation.vector;
%         x = -r'*t;
%         AllPosesComputed(frame,:) = x';
%     end
    AllPosesComputed(:,3) = abs(AllPosesComputed(:,3));
    
    %% % Plot the last two figures (POST-GTSAM)
    
    figure(3);

    if plotPoints
        plot3(AllPosesComputed(:,1),AllPosesComputed(:,2),AllPosesComputed(:,3),'o');
        hold on;
        title('DatasetwithGTSAM-no pose');
        plot3(LandMarksComputed(:,2),LandMarksComputed(:,3), zeros(81,1), 'g*');
        hold off;
        legend('show')
    end
    
    figure(4);
    if plotPoints
        plot3(AllPosesComputed(:,1),AllPosesComputed(:,2),AllPosesComputed(:,3),'o');
        hold on;
        title('DatasetwithGTSAM');
        plot3(LandMarksComputed(:,2),LandMarksComputed(:,3), LandMarksComputed(:,4), 'r*');
        plot3(LandMarksComputed(:,5),LandMarksComputed(:,6), LandMarksComputed(:,7),'b*');
        plot3(LandMarksComputed(:,8),LandMarksComputed(:,9), LandMarksComputed(:,10),'green*');
        plot3(LandMarksComputed(:,11),LandMarksComputed(:,12), LandMarksComputed(:,13),'black*');
        
        hold off;
        legend('x','y','z')
    end
 end
 
function H = est_homography(X,Y,x,y)
% Code Reference: https://www.mathworks.com/matlabcentral/answers/26141-homography-matrix
    A = zeros(length(x(:))*2,9);

    for i = 1:length(x(:))
        a = [x(i),y(i),1];
        A((i-1)*2+1:(i-1)*2+2,1:9) = [[[x(i),y(i),1] [0 0 0];[0 0 0] [x(i),y(i),1]] -[X(i);Y(i)]*a];
    end

    [U S V] = svd(A);
    H = reshape(V(:,9),3,3)';
end 
