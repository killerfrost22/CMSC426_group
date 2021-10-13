function [ransac_matchedPoints1, ransac_matchedPoints2, H_best] = RANSAC(matchedPoints1, matchedPoints2, curr_img_descriptors, next_img_descriptors, best_matches, curr_img_x_best, curr_img_y_best, next_img_x_best, next_img_y_best)

% matchedPoints1, matchedPoints2, curr_img_descriptors, next_img_descriptors, best_matches, 
% curr_img_x_best, curr_img_y_best, next_img_x_best, next_img_y_best)

% 1. Select four pairs of matched feature descriptors (at random), 1<=i<=4, p_1i and p_2i from images 1/2.
% 2. Compute the homography matrix from the four feature pairs using est_homography
% 3. Compute inliers where SSD(p_2, Hp_1) < threshold
% 4. Repeat the last 3 steps until you reach N_max iters or 90% of inliers.
% 5. Keep largest set of inliers.
% 6. Recompute least-square H on all inliers.
end