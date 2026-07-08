function plc_correction(folderA, slopefile, aspectfile, folderB)
% PLC地形校正函数
% 输入：
%   folderA    - Landsat SR影像文件夹
%   slopefile  - 坡度文件路径
%   aspectfile - 坡向文件路径
%   folderB    - 输出文件夹

% 并行设置
myCluster = parcluster('local');
maxWorkers = myCluster.NumWorkers;
numWorkersToStart = max(maxWorkers - 2, 1);
existingPool = gcp('nocreate');
if isempty(existingPool)
    parpool(myCluster, numWorkersToStart);
elseif existingPool.NumWorkers ~= numWorkersToStart
    delete(existingPool);
    parpool(myCluster, numWorkersToStart);
end
fprintf('最大可用 workers: %d, 实际启动: %d\n', maxWorkers, numWorkersToStart);

% 基本参数
rd = pi/180;
rdt = 180/pi;

% 读取坡度与坡向
slopeData = double(imread(slopefile));
aspectData = double(imread(aspectfile));

% 参考数据
filesA = dir(fullfile(folderA,'*.tif'));
if isempty(filesA)
    error('SR影像文件夹为空: %s', folderA);
end

[~, ref_R] = readgeoraster(fullfile(folderA, filesA(1).name));
ref_info = geotiffinfo(fullfile(folderA, filesA(1).name));

if ~exist(folderB,'dir'), mkdir(folderB); end

% 逐影像处理（保留原有逻辑）
parfor i = 1:length(filesA)
    try
        fname = filesA(i).name;
        fpath = fullfile(folderA, fname);
        outpath = fullfile(folderB, fname);

        if isfile(outpath), continue; end

        img = readgeoraster(fpath);
        out_img = zeros(size(img), 'double') + NaN;

        [rowv, colv] = find(~isnan(img(:,:,1)), 1);
        tphi = double(img(rowv, colv, 8));

        if size(rowv, 1) == 0
            fprintf('%s 影像全为NaN，跳过！\n', fname);
            continue;
        end

        [row, col] = find(img(:,:,1));

        for j = 1:length(row)
            ValidValue = double(reshape(img(row(j),col(j),:), 1, []));
            Slope = double(slopeData(row(j), col(j)));
            Aspect = double(aspectData(row(j), col(j)));
            ts = ValidValue(7);

            o = acos(cos(ts*rd)*cos(Slope*rd) + ...
                sin(ts*rd)*sin(Slope*rd)*cos((tphi-Aspect)*rd)) * rdt;

            if o >= 89.999 || any(ValidValue(1:6) < 0)
                continue;
            end

            lft = 1/cos(ts*rd);
            lvt = 1/(cos(ts*rd)*(1-tan(Slope*rd)*cos((tphi-Aspect)*rd)*tan(ts*rd)));
            lfv = 1;
            lvv = 1;

            sr_tc = ValidValue(1:6).*(lft+lfv)./(lvt+lvv);
            all = [sr_tc, ValidValue(7), ValidValue(8)];
            out_img(row(j), col(j), :) = reshape(all, [1,1,size(img,3)]);
        end

        geotiffwrite(outpath, out_img, ref_R, "CoordRefSysCode", ref_info.GeoTIFFCodes.PCS);
        fprintf('%s 处理成功！\n', fname);

    catch ME
        fprintf('处理 %s 出错：%s\n', fname, ME.message);
        continue;
    end
end

% 关闭并行池
poolobj = gcp('nocreate');
if ~isempty(poolobj), delete(poolobj); end

fprintf('PLC地形校正全部完成！\n');
end

