clear;
clc;
%% Load files
load('sun/sun_256_for_check.mat');
B = binary_codes;
label = lab(:,1);

%% Category retrieval
B = sign(B-0.5);
[~, ~, map] = compute_map(-B*B', label, label, true);
fprintf('Retrieval mAP of category retrieval: %.4f\n', map);

%% Category retrieval with expert-bit
top_k_bits = 12;
dis_mtx = select_distance(B,B,label,GRM,top_k_bits); % top-12 bits selected
[~,~,map_e]=compute_map(dis_mtx,label,label,true);
fprintf('Retrieval mAP of category retrieval using expert bits: %.4f\n', map_e);

%% Attribute retrieval
target_attribute_idx = [15,41,42,43,66,67,69];
attribute_test = lab(:, 2:end);
map_a = compute_map_attribute_rank(B, B, attribute_test, attribute_test);
fprintf('Retrieval mAP of attribute retrieval: %.4f\n', mean(map_a(target_attribute_idx)));

%% For attribute retrieval using customized bits
expert_idx={[69, 78] [76] [33, 109] [89] [2,123,4] [15, 65] [112,104,105]}'; % each cell record the attribute-related interpreted bit index
compute_map_attribute_rank(B, B, attribute_test(:,41), attribute_test(:,41), expert_idx);
