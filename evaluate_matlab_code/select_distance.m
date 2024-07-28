function [ dis_mtx ] = select_distance(q_B,d_B, q_labels, GRM, topN)
%   输入查询和DB的二值码，查询的类别标注，以及类别规则矩阵，每个查询只选择最相关的比特和DB中样本对应的比特计算海明距离矩阵

q_num = size(q_B,1);
d_num = size(d_B,1);
lab_num = size(GRM,1);

% topN = 200; % 尝试不同的数值

dis_mtx = zeros(d_num, q_num);

for lab = 1:lab_num
    rule = GRM(lab,:);
    [~, idx] = sort(rule, 'descend');
    
    tmp_d_B = d_B(:, idx(1:topN));
    
    idx2 = q_labels == lab-1;
    if sum(idx2)==0
        continue;
    end
    
    tmp_q_B = q_B(idx2, idx(1:topN));
    
    tmp_dismtx = -tmp_d_B*tmp_q_B';
    dis_mtx(:, idx2) = tmp_dismtx;
end

end

