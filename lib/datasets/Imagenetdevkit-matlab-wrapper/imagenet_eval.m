function res = imagenet_eval(path, comp_id, test_set, output_dir, rm_res)

% this script demos the usage of evaluation routines for detection task
% the result file 'demo.val.pred.det.txt' on validation data is evaluated
% against the ground truth

% DJDJ
%{
path = '/home/dj/big/data/ilsvrc14/ILSVRC2015_devkit/evaluation'
comp_id ='comp4-1096'
test_set = 'val'
output_dir = '/home/dj/big/workspace/fast-rcnn/output/fast_rcnn/imagnet_val/vgg_cnn_m_1024_imagenet_fast_rcnn_step_2_iter_40000_with_step_1_rpn_top_2300'
rm_res = 1
%}


%warning ("off", "Octave:num-to-str");
%warning ("off", "Octave:possible-matlab-short-circuit-operator");

fprintf('DETECTION TASK\n');

org_dir = pwd;
cd(path);

%pred_file='demo.val.pred.det.txt';
meta_file = '../data/meta_det.mat';
if test_set == 'val_2000'
    eval_file = '../data/det_lists/val_2000.txt';
    optional_cache_file = '../data/ILSVRC2015_det_validation_ground_truth_2000.mat';
else
    eval_file = '../data/det_lists/val.txt';
    optional_cache_file = '../data/ILSVRC2015_det_validation_ground_truth.mat';
end

blacklist_file = '../data/ILSVRC2015_det_validation_blacklist.txt';

pred_file = sprintf('%s/../results/%s_det_%s.txt', path, comp_id, test_set)

ground_truth_dir = '/home/dj/big/data/ilsvrc14/ILSVRC2013_DET_bbox_val';

fprintf('pred_file: %s\n', pred_file);
fprintf('meta_file: %s\n', meta_file);
fprintf('eval_file: %s\n', eval_file);
fprintf('blacklist_file: %s\n', blacklist_file);

if isempty(optional_cache_file)
    fprintf(['NOTE: you can specify a cache filename and the ground ' ...
             'truth data will be automatically cached to save loading time ' ...
             'in the future\n']);
end

num_val_files = -1;
while num_val_files ~= 20121
    if num_val_files ~= -1
        fprintf('That does not seem to be the correct directory. Please try again\n');
    end
    %ground_truth_dir = input(['Please enter the path to the Validation bounding box ' ...
    %               'annotations directory: '],'s');
    val_files = dir(sprintf('%s/*val*.xml',ground_truth_dir));
    num_val_files = numel(val_files);
end

[ap recall precision] = eval_detection(pred_file,ground_truth_dir,meta_file,eval_file,blacklist_file,optional_cache_file);

load(meta_file);
fprintf('-------------\n');
fprintf('Category\tAP\n');
for i=[1:5 196:200]
    s = synsets(i).name;
    if length(s) < 8
        fprintf('%s\t\t%0.3f\n',s,ap(i));
    else
        fprintf('%s\t%0.3f\n',s,ap(i));
    end
    if i == 5
        fprintf(' ... (190 categories)\n');
    end
end
fprintf(' - - - - - - - - \n');
fprintf('Mean AP:\t %0.3f\n',mean(ap));
fprintf('Median AP:\t %0.3f\n',median(ap));

output_file = sprintf('%s/result_%s.txt', output_dir, comp_id);
fileID = fopen(output_file,'w');
fprintf(fileID, '\n~~~~~~~~~~~~~~~~~~~~\n');
fprintf(fileID, 'Results:\n');
fprintf(fileID, '%.1f\n', ap * 100);
fprintf(fileID, '%.1f\n', mean(ap) * 100);
fprintf(fileID, '~~~~~~~~~~~~~~~~~~~~\n');
fclose(fileID);


%input('enter to continue'); 


cd(org_dir);