function [ map ] = compute_map_attribute_rank(q_B, d_B, q_label, d_label, expert_idx)
% 单独考察每个正例属性的检索效果，可以基于规则矩阵筛选对应的比特进行海明距离计算

% q_num = size(q_B, 1);
% d_num = size(d_B, 1);
lab_num = size(q_label, 2);
map = zeros(lab_num, 1);

% topN = 5; % 需要调整

flag = false;
if nargin == 5
    flag = true;
    num_ele = length(expert_idx);
    choice_num = 0;
    for i = 1:num_ele
        choice_num = choice_num + nchoosek(num_ele, i);
    end
    choice = cell(choice_num, 1);
    count = 1;
    for i = 1:num_ele
        temp = nchoosek(1:num_ele, i);
        for j=1:size(temp,1)
            choice{count} = temp(j,:);
            count = count + 1;
        end
    end
    
end

best_choice = [];
rand_idx = randperm(size(q_B,2));
best_choice_full1 = [];
for lab = 1:lab_num
    idx = q_label(:, lab) == 1;
    best_map = 0.;
    best_rand_map = 0;
    best_full_1_map = 0;
    best_full_1_rand_map = 0;
    if flag 
%         rule = GRM(lab,:);
%         [~, idx1] = sort(rule, 'descend');
%         tmp_d_B = d_B(:, idx1(1:topN));
%         tmp_q_B = q_B(idx, idx1(1:topN));
        
        for k=1:size(choice,1)
            idx1 = [];
            for m = 1:length(choice{k})
                idx1 = [idx1, expert_idx{choice{k}(m)}];
            end
            tmp_d_B = d_B(:, idx1);
            tmp_q_B = q_B(idx, idx1);
            [~, ~, cur_map] = compute_map(-tmp_d_B*tmp_q_B', q_label(idx, lab), d_label(:, lab), false);
            
            if cur_map>best_map
                best_map = cur_map;
                best_choice = choice{k};
                [~,~,best_rand_map] = compute_map(-d_B(:, rand_idx(1:length(idx1)))*q_B(idx, rand_idx(1:length(idx1)))', q_label(idx, lab), d_label(:, lab), false);
            end
            tmp_q_B = ones(1, size(tmp_q_B, 2));
            [~,~, cur_map_full1] =  compute_map(-tmp_d_B*tmp_q_B', ones(1,1), d_label(:, lab), false);
            if cur_map_full1 > best_full_1_map
                best_full_1_map = cur_map_full1;
                best_choice_full1 = choice{k};
                [~,~,best_full_1_rand_map] = compute_map(-d_B(:, rand_idx(1:length(idx1)))*tmp_q_B', ones(1,1), d_label(:, lab), false);
            end
        end
        map(lab) = best_map;
    else
        tmp_q_B = q_B(idx, :);
        tmp_d_B = d_B;
        [~, ~, map(lab)] = compute_map(-tmp_d_B*tmp_q_B', q_label(idx, lab), d_label(:, lab), false);
    end
    
end

% map = mean(map);
if nargin == 5
    best_choice
    best_choice_full1
    best_rand_map
    best_full_1_map
    best_full_1_rand_map
end
end

