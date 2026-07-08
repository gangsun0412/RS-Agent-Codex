function rf_train(landsat_folder, glass_folder, model_save_path)
% 随机森林训练函数
% 输入：
%   landsat_folder   - PLC校正后Landsat SR文件夹（7波段，取前7波段）
%   glass_folder     - 预处理后GLASS NDVI文件夹
%   model_save_path  - 模型保存完整路径（含文件名，如D:\model\RF_model.mat）

% 并行设置
myCluster = parcluster('local');
maxWorkers = myCluster.NumWorkers;
numWorkersToStart = max(maxWorkers-2, 1);
existingPool = gcp('nocreate');
if isempty(existingPool)
    parpool(myCluster, numWorkersToStart);
elseif existingPool.NumWorkers ~= numWorkersToStart
    delete(existingPool);
    parpool(myCluster, numWorkersToStart);
end
fprintf('最大可用 workers: %d, 实际启动: %d\n', maxWorkers, numWorkersToStart);

t1 = dir(fullfile(landsat_folder, '*.tif'));
t2 = dir(fullfile(glass_folder, '*.tif'));

if isempty(t1), error('Landsat文件夹为空: %s', landsat_folder); end
if isempty(t2), error('GLASS文件夹为空: %s', glass_folder); end

num = 1:8:365;
ff = cell(length(t1), 1);

[L_row, L_col, ~] = size(imread(fullfile(landsat_folder, t1(1).name)));
G_row = floor(L_row / 8);
G_col = floor(L_col / 8);
L_row = G_row * 8;
L_col = G_col * 8;

% 数据处理
for i = 1:length(t1)
    fn = t1(i).name;
    fpath = fullfile(landsat_folder, fn);

    % 解析DOY
    doy = day(datetime(string(fn(1:8)), "InputFormat", "yyyyMMdd"), 'dayofyear');
    inx = find(abs(num - doy) == min(abs(num - doy)));
    if isempty(inx), continue; end

    % 匹配GLASS文件
    t2_match = dir(fullfile(glass_folder, [fn(1:4), '*', num2str(num(inx(1)), '%3.3i'), '*.tif']));
    if isempty(t2_match), continue; end

    fname2 = fullfile(glass_folder, t2_match(1).name);
    if ~isfile(fname2), continue; end

    % 读取数据（取前7波段）
    a = imresize(double(imread(fpath)), [L_row, L_col]);
    a = a(:, :, 1:7);  % 只取前7波段
    b = imresize(double(imread(fname2)), [G_row, G_col]);

    % 8x8降采样
    result = zeros(G_row, G_col, 7);
    for i2 = 1:G_row
        for j = 1:G_col
            for k = 1:7
                block = a((i2-1)*8+1:i2*8, (j-1)*8+1:j*8, k);
                block(block == 0) = NaN;
                result(i2, j, k) = nanmean(block(:));
            end
        end
    end

    % 重塑为样本矩阵
    r2 = zeros(G_row * G_col, 7);
    for i3 = 1:7
        r2(:, i3) = reshape(result(:, :, i3), [], 1);
    end
    b = reshape(b, [], 1);

    % 过滤无效值
    ab = [r2, b];
    ab = ab(ab(:, 1) > 0 & ab(:, end) > 0, :);
    ff{i} = ab;
end

% 合并训练数据
fin = [];
for i = 1:length(ff)
    fin = [fin; ff{i}];
end
fin(:, end) = fin(:, end) * 0.0001;
clear ff

fprintf('数据处理完成，开始随机森林训练！\n');

% 随机森林训练
rowrank = randperm(size(fin, 1));
fin = fin(rowrank, :);
nTree = 50;

input_train = fin(1:round(size(fin,1)*0.85), 1:end-1);
output_train = fin(1:round(size(fin,1)*0.85), end);

paroptions = statset('UseParallel', true);
Factor = TreeBagger(nTree, input_train, output_train, ...
    'Method', 'regression', 'Options', paroptions);
clear fin

% 保存模型
model_dir = fileparts(model_save_path);
if ~exist(model_dir, 'dir'), mkdir(model_dir); end
save(model_save_path, 'Factor');
fprintf('模型已保存至：%s\n', model_save_path);

% 关闭并行池
poolobj = gcp('nocreate');
if ~isempty(poolobj), delete(poolobj); end

fprintf('随机森林训练完成！\n');
end