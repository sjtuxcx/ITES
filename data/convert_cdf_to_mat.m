

pose_directory = 'pose';
dirs = dir(strcat(pose_directory, '/*/MyPoseFeatures/D3_Positions/*.cdf'));

paths = {dirs.folder};
names = {dirs.name};

for i = 1:numel(names)
    data = cdfread(strcat(paths{i}, '/', names{i}));
    save(strcat(paths{i}, '/', names{i}, '.mat'), 'data');
end