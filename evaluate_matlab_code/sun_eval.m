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
%target_attribute_idx = [15,41,42,43,66,67,69];
attribute_test = lab(:, 2:end);
map_a = compute_map_attribute_rank(B, B, attribute_test, attribute_test);
fprintf('Retrieval mAP of ocean attribute retrieval: %.4f\n', map_a(67));

%% For attribute retrieval using customized bits
% ocean related interpreted filters' bit index: [water], [sea], [skycraper], [water tower], [jacuzzi-indoor] 
expert_idx={[180, 39, 101, 79] [223, 75] [35, 250, 205, 82, 19, 70, 111, 251] [237, 256, 20, 48] [233, 23, 240]}'; % each cell record the attribute-related (grass) interpreted bit index
map_custom = compute_map_attribute_rank(B, B, attribute_test(:,67), attribute_test(:,67), expert_idx);
fprintf('Retrieval mAP of ocean attribute customized retrieval: %.4f\n', map_custom);
