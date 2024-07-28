clear;
clc;
%% Load files
load('imagenet/imagenet_256_for_check.mat');
load('../data/ImageNet/attr_w.mat');
B = binary_codes;
label = lab(:,1);

%% Category retrieval
B = sign(B-0.5);
[~, ~, map] = compute_map(-B*B', label, label, true);
fprintf('Retrieval mAP of category retrieval: %.4f\n', map);

%% Category retrieval with expert-bit
top_k_bits = 13;
dis_mtx = select_distance(B,B,label,GRM,top_k_bits); % top-12 bits selected
[~,~,map_e]=compute_map(dis_mtx,label,label,true);
fprintf('Retrieval mAP of category retrieval using expert bits: %.4f\n', map_e);
