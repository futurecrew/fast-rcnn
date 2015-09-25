function Imagenetopts = get_imagenet_opts(path)

tmp = pwd;
cd(path);
try
  addpath('evaluation');
catch
  rmpath('evaluation');
  cd(tmp);
  error(sprintf('evaluation directory not found under %s', path));
end
rmpath('evaluation');
cd(tmp);
