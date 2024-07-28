function [topN_idx, topN, map] = compute_map(dis_mtx, query_label, database_label, same_train_test)
% 输入查询和DB的海明距离矩阵，查询的类别标签，DB的类别标签，基于是否同类计算mAP
%tic;

if nargin < 4
    same_train_test = false;
end

q_num = length(query_label);
d_num = length(database_label);
map = zeros(q_num, 1);

database_label_mtx = repmat(database_label, 1, q_num);
if same_train_test
    database_label_mtx = database_label_mtx.*(1-eye(d_num));
    database_label_mtx = database_label_mtx - eye(d_num);
end
sorted_database_label_mtx = database_label_mtx;

[mtx idx_mtx] = sort(dis_mtx, 1);

for q = 1 : q_num
    sorted_database_label_mtx(:, q) = database_label_mtx(idx_mtx(:, q), q);
end

result_mtx = (sorted_database_label_mtx == repmat(query_label', d_num, 1));

% topN precision recall and F1 score
N_ = 100;
topN = sum(result_mtx(1:N_, :));
% precision = mean(topN ./ N_);
% fprintf('precision is %.4f:\n', precision);
% total_N = sum(result_mtx(:,:));
% recall = mean(topN./total_N);
% fprintf('recall is %.4f:\n', recall);
topN_idx = idx_mtx(1:N_, :);
% F_score = 2*precision*recall/(precision+recall+1e-12);
% fprintf('F1 score is %.4f:\n', F_score);

for q = 1 : q_num
    Qi = sum(result_mtx(:, q));
    if Qi > 0
        map(q) = sum( ([1:Qi]') ./ (find(result_mtx(:, q) == 1)) ) / Qi;
    else
        map(q) = 0;
    end                
end

map = mean(map);
% toc;
end