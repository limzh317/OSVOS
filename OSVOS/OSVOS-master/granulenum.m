fileFolder=fullfile('/home/zeven/桌面/cell/countnum');
 
dirOutput=dir(fullfile(fileFolder,'*.mat'));
 
fileNames={dirOutput.name};
t = transpose(fileNames);
i=1;
output =[];
for i = 1:42
    bin_mask_file = [];
    name = ['/home/zeven/桌面/cell/countnum/' t{i}];
    i
    load(name)
    bin_mask_file(1);
    bin = bin_mask_file;
    bin4 = bin_mask_file;
    bin(bin==3) = 0;
    bin4=(bin4==4);
    bin=(bin>0);
    gran_voxel = sum(bin4(:));
    cystol_num = sum(bin(:));
    AA = bwconncomp(bin4,26);
    gran_num = AA.NumObjects;
    output(i,1) = gran_num;
    output(i,2) = gran_voxel;
    output(i,3) = cystol_num;
end
