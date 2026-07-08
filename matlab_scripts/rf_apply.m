function rf_apply(landsat_folder, model_path, output_folder)
% 随机森林应用函数
% 输入：
%   landsat_folder  - PLC校正后Landsat SR文件夹
%   model_path      - 训练好的模型路径（.mat文件）
%   output_folder   - 输出文件夹

start_time = tic;

if ~exist(output_folder, 'dir'), mkdir(output_folder); end

% 加载模型
if ~isfile(model_path)
    error('模型文件不存在: %s', model_path);
end
load(model_path, 'Factor');
fprintf('模型加载完成！\n');

% 读取文件列表
t1 = dir(fullfile(landsat_folder, '*.tif'));
if isempty(t1), error('Landsat文件夹为空: %s', landsat_folder); end

% 读取地理信息
[~, R] = readgeoraster(fullfile(landsat_folder, t1(1).name));
info = geotiffinfo(fullfile(landsat_folder, t1(1).name));

% 并行应用模型
parfor i = 1:length(t1)
    fn = t1(i).name;
    output_path = fullfile(output_folder, fn);
    if isfile(output_path), continue; end

    a = double(imread(fullfile(landsat_folder, fn)));
    a = a(:, :, 1:7);  % 只取前7波段
    [L_row, L_col, L_hei] = size(a);
    numPixels = L_row * L_col;

    % NaN掩膜（0值无意义，设为NaN）
    valid_nan = any(a == 0, 3);
    a(repmat(valid_nan, [1, 1, L_hei])) = NaN;

    % 数据重塑
    result = reshape(a, numPixels, []);
    result(:, 1:6) = result(:, 1:6) * 0.0001;
    result(:, 7) = rad2deg(acos(result(:, 7) * 0.0001));

    % 过滤复数（acos超出范围产生）
    valid_rows = any(imag(result) ~= 0, 2);
    result(valid_rows, :) = NaN;

    % 模型估算
    predictedValues = predict(Factor, result);
    predictedImage = reshape(predictedValues, [L_row, L_col]);
    predictedImage(valid_nan) = NaN;

    % 写出结果
    geotiffwrite(output_path, predictedImage, R, ...
        'CoordRefSysCode', info.GeoTIFFCodes.PCS);
    fprintf('%s 估算完成！\n', fn);
end

elapsed_time = toc(start_time);
fprintf('全部估算完成，耗时：%.4f秒！\n', elapsed_time);

% 关闭并行池
poolobj = gcp('nocreate');
if ~isempty(poolobj), delete(poolobj); end
end