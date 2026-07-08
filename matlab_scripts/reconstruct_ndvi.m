function reconstruct_ndvi(landsat_folder, glass_folder, output_folder, parameter_folder)
% NDVI时序重建函数（两阶段：16天→8天）
% 输入：
%   landsat_folder   - RF估算后的Landsat NDVI文件夹（[-1,1] double）
%   glass_folder     - 预处理后GLASS NDVI文件夹（0-10000 uint16）
%   output_folder    - 最终输出文件夹
%   parameter_folder - 回归参数矩阵保存文件夹

total_start_time = tic;

if ~exist(output_folder, 'dir'), mkdir(output_folder); end
if ~exist(parameter_folder, 'dir'), mkdir(parameter_folder); end

% 中间结果文件夹
temp_folder = fullfile(output_folder, 'temp_16d');
if ~exist(temp_folder, 'dir'), mkdir(temp_folder); end

% 并行设置
myCluster = parcluster('local');
myCluster.NumWorkers = 6;
if isempty(gcp('nocreate'))
    parpool(myCluster, 6);
end

%% ====================两阶段处理====================
rebuild_8d = false;

for stage = 1:2

    if rebuild_8d == false
        landsat_datadir = [landsat_folder, '\'];
        target_datadir  = [temp_folder, '\'];
        fprintf('------ 第一阶段：16天重建开始 ------\n');
    else
        landsat_datadir = [temp_folder, '\'];
        target_datadir  = [output_folder, '\'];
        fprintf('------ 第二阶段：8天序列生成开始 ------\n');
    end

    GLASS_datadir = [glass_folder, '\'];
    if ~exist(target_datadir, 'dir'), mkdir(target_datadir); end

    %% ====================读取文件====================
    L2_file = dir(fullfile(landsat_datadir, '*.tif'));
    M2_file = dir(fullfile(GLASS_datadir, '*.tif'));

    if isempty(L2_file), error('Landsat文件夹为空: %s', landsat_datadir); end
    if isempty(M2_file), error('GLASS文件夹为空: %s', GLASS_datadir); end

    %% ====================时间匹配====================
    L2_dates = datetime(extractBefore({L2_file.name}, 9), 'InputFormat', 'yyyyMMdd');
    M2_dates = datetime(extractBefore({M2_file.name}, 9), 'InputFormat', 'yyyyMMdd');
    date_diff_matrix = days(abs(L2_dates' - M2_dates));
    fprintf('时间匹配完成，开始重建！\n');

    %% ====================预加载GLASS数据====================
    M2_len = length(M2_file);
    L2_len = length(L2_file);
    M2_data_cache = cell(M2_len, 1);

    parfor j = 1:M2_len
        M2_path = fullfile(GLASS_datadir, M2_file(j).name);
        M2_data_cache{j} = double(imread(M2_path)) / 10000;
    end

    %% ====================计算回归参数矩阵====================
    Parameter_name = 'Parameter_reconstruct.mat';
    Parameter_mat  = fullfile(parameter_folder, Parameter_name);

    if exist(Parameter_mat, 'file')
        load(Parameter_mat, 'ssim_matrix', 'R2_matrix', 'slope_matrix', 'intercept_matrix');
        fprintf('成功加载回归参数矩阵！\n');
    else
        [ssim_matrix, R2_matrix, slope_matrix, intercept_matrix] = deal(zeros(M2_len));

        parfor j = 1:M2_len
            M2_data = M2_data_cache{j};
            for k = 1:M2_len
                if j == k, continue; end
                M1_data = M2_data_cache{k};
                valid = ~isnan(M1_data) & ~isnan(M2_data);
                X = M1_data(valid); Y = M2_data(valid);
                X_design = [ones(size(X)), X];
                coeff = X_design \ Y;
                y_pred = X_design * coeff;
                R2_matrix(j,k) = 1 - sum((Y - y_pred).^2) / sum((Y - mean(Y)).^2);
                slope_matrix(j,k) = coeff(2);
                intercept_matrix(j,k) = coeff(1);
            end
        end

        save(Parameter_mat, 'ssim_matrix', 'R2_matrix', 'slope_matrix', 'intercept_matrix');
        fprintf('回归参数矩阵计算完成并保存！\n');
    end

    %% ====================主处理循环====================
    L2_names      = {L2_file.name};
    outputFilenames = fullfile(target_datadir, L2_names);

    parfor i = 1:L2_len
        current_L2_name = L2_names{i};
        L2_path = fullfile(landsat_datadir, current_L2_name);
        [L2_data, R] = geotiffread(L2_path);
        info = geotiffinfo(L2_path);
        L2_data = double(L2_data);

        % 无NaN直接写出
        if all(~isnan(L2_data(:)))
            geotiffwrite(outputFilenames{i}, L2_data, R, ...
                'CoordRefSysCode', info.GeoTIFFCodes.PCS);
            continue;
        end

        % 寻找最近GLASS日期
        [~, closest_M2_idx] = min(date_diff_matrix(i,:));

        % 筛选R2>0.7的候选
        related_R2 = R2_matrix(closest_M2_idx, :);
        [sorted_R2, sort_order] = sort(related_R2, 'descend');
        candidate_indices = sort_order(sorted_R2 >= 0.7);

        % 迭代填充NaN
        L2_NaN_mask = isnan(L2_data);
        update_data = L2_data;
        used_dates  = NaT(0);
        fill_progress = [];

        for k = 1:length(candidate_indices)
            M1_idx        = candidate_indices(k);
            date_diffs    = date_diff_matrix(:, M1_idx);
            date_diffs(i) = Inf;

            [~, closest_L1_idx]  = min(date_diffs);
            closest_L1_date      = L2_dates(closest_L1_idx);
            closest_L1_name      = L2_names{closest_L1_idx};

            if any(closest_L1_date == used_dates), continue; end

            L1_path = fullfile(landsat_datadir, closest_L1_name);
            if ~isfile(L1_path), continue; end

            L1_data = double(imread(L1_path));
            if all(isnan(L1_data(:))), continue; end

            slope     = slope_matrix(closest_M2_idx, M1_idx);
            intercept = intercept_matrix(closest_M2_idx, M1_idx);
            corrected_data = L1_data * slope + intercept;

            update_data(L2_NaN_mask) = corrected_data(L2_NaN_mask);
            used_dates(end+1)  = closest_L1_date;
            remaining_nan      = nnz(isnan(update_data));
            fill_progress      = [fill_progress; remaining_nan];

            if remaining_nan == 0
                break;
            else
                L2_NaN_mask = isnan(update_data);
            end
        end

        % 写出结果
        if ~isempty(fill_progress) && fill_progress(end) < nnz(isnan(L2_data))
            geotiffwrite(outputFilenames{i}, int32(update_data * 10000), R, ...
                'CoordRefSysCode', info.GeoTIFFCodes.PCS);
        else
            geotiffwrite(outputFilenames{i}, int32(L2_data * 10000), R, ...
                'CoordRefSysCode', info.GeoTIFFCodes.PCS);
        end
    end

    %% ====================生成8天序列====================
    if rebuild_8d == false
        fprintf('16天重建完成！开始生成8天序列...\n');

        L2_dates_list1 = L2_dates(2):caldays(8):L2_dates(end);
        L2_dates_list2 = sort((L2_dates(2) + caldays(-8)):caldays(-8):L2_dates(1));
        L2_dates_list  = [L2_dates(1), L2_dates_list2, L2_dates_list1];

        [L2_data, R] = geotiffread(fullfile(landsat_datadir, L2_file(1).name));
        info = geotiffinfo(fullfile(landsat_datadir, L2_file(1).name));
        NaN_image = nan(size(L2_data), 'double');

        parfor n = 1:length(L2_dates_list)
            if ismember(L2_dates_list(n), L2_dates), continue; end
            outputFile = fullfile(target_datadir, ...
                [datestr(L2_dates_list(n), 'yyyymmdd'), '.tif']);
            geotiffwrite(outputFile, NaN_image, R, ...
                'CoordRefSysCode', info.GeoTIFFCodes.PCS);
        end

        rebuild_8d = true;

    else
        fprintf('8天序列生成完成！\n');
        break;
    end
end

total_elapsed_time = toc(total_start_time);
fprintf('重建全部完成！总耗时：%.4f秒\n', total_elapsed_time);

poolobj = gcp('nocreate');
if ~isempty(poolobj), delete(poolobj); end
end