function res = imagenet_eval(path, comp_id, test_set, output_dir, rm_res)

% this script demos the usage of evaluation routines for detection task
% the result file 'demo.val.pred.det.txt' on validation data is evaluated
% against the ground truth

% DJDJ
warning ("off", "Octave:num-to-str");
warning ("off", "Octave:possible-matlab-short-circuit-operator");

fprintf('DETECTION TASK\n');

get_imagenet_opts(path);

pred_file='demo.val.pred.det.txt';
meta_file = '../data/meta_det.mat';
eval_file = '../data/det_lists/val.txt';
blacklist_file = '../data/ILSVRC2014_det_validation_blacklist.txt';

optional_cache_file = '';
gtruth_directory = 'D:/data/ilsvrc14/ILSVRC2013_DET_bbox_val';

fprintf('pred_file: %s\n', pred_file);
fprintf('meta_file: %s\n', meta_file);
fprintf('eval_file: %s\n', eval_file);
fprintf('blacklist_file: %s\n', blacklist_file);
if isempty(optional_cache_file)
    fprintf(['NOTE: you can specify a cache filename and the ground ' ...
             'truth data will be automatically cached to save loading time ' ...
             'in the future\n']);
end

while isempty(gtruth_directory)
    g_dir = input(['Please enter the path to the Validation bounding box ' ...
                   'annotations directory: '],'s');
    d = dir(sprintf('%s/*val*.xml',g_dir));
    if length(d) == 0
        fprintf(['does not seem to be the correct directory, please ' ...
                 'try again\n']);
    else
        gtruth_directory = g_dir;
    end
end

[ap recall precision] = eval_detection(pred_file,gtruth_directory,meta_file,eval_file,blacklist_file,optional_cache_file);

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


