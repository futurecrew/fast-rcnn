function res = voc_eval(path, comp_id, test_set, output_dir, rm_res)

% DJDJ
%{
path = 'E:/data/VOCdevkit2';
comp_id = 'comp4-6060';
test_set = 'test';
output_dir = 'E:\project\fast-rcnn\output\fast_rcnn\voc_2007_test\vgg_cnn_m_1024_fast_rcnn_iter_40000_with_step_1_rpn_top_2300';
rm_res = 1;
%}

VOCopts = get_voc_opts(path);
VOCopts.testset = test_set;

for i = 1:length(VOCopts.classes)
  cls = VOCopts.classes{i};
  res(i) = voc_eval_cls(cls, VOCopts, comp_id, output_dir, rm_res);
end

fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Results:\n');
aps = [res(:).ap]';
fprintf('%.1f\n', aps * 100);
fprintf('%.1f\n', mean(aps) * 100);
fprintf('~~~~~~~~~~~~~~~~~~~~\n');

output_file = sprintf('%s/result_%s.txt', output_dir, comp_id);
fileID = fopen(output_file,'w');
fprintf(fileID, '\n~~~~~~~~~~~~~~~~~~~~\n');
fprintf(fileID, 'Results:\n');
fprintf(fileID, '%.1f\n', aps * 100);
fprintf(fileID, '%.1f\n', mean(aps) * 100);
fprintf(fileID, '~~~~~~~~~~~~~~~~~~~~\n');
fclose(fileID);


input('enter to continue'); 

function res = voc_eval_cls(cls, VOCopts, comp_id, output_dir, rm_res)

test_set = VOCopts.testset;
year = VOCopts.dataset(4:end);

addpath(fullfile(VOCopts.datadir, 'VOCcode'));

res_fn = sprintf(VOCopts.detrespath, comp_id, cls);

recall = [];
prec = [];
ap = 0;
ap_auc = 0;

do_eval = (str2num(year) <= 2007) | ~strcmp(test_set, 'test');
if do_eval
  % Bug in VOCevaldet requires that tic has been called first
  tic;
  [recall, prec, ap] = VOCevaldet(VOCopts, comp_id, cls, true);
  ap_auc = xVOCap(recall, prec);

  % force plot limits
  ylim([0 1]);
  xlim([0 1]);

  print(gcf, '-djpeg', '-r0', ...
        [output_dir '/' cls '_pr.jpg']);
end
fprintf('!!! %s : %.4f %.4f\n', cls, ap, ap_auc);

res.recall = recall;
res.prec = prec;
res.ap = ap;
res.ap_auc = ap_auc;

save([output_dir '/' cls '_pr.mat'], ...
     'res', 'recall', 'prec', 'ap', 'ap_auc');

if rm_res
  delete(res_fn);
end

rmpath(fullfile(VOCopts.datadir, 'VOCcode'));
