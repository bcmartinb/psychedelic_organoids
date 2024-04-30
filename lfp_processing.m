filename = '/Users/blancamartin/Desktop/LFP_org_psychedelics/2024-03-20_LFP_10min_no_stim_during_recording_was_stim_previously/118-5923/5MeOtreated-CheRiff-20032024-10minLFP(000)_BroadbandProcessor.raw';

% Load the low frequency broadband from .raw file
data = AxisFile(filename).RawVoltageData;
lfp_data = data(2).LoadData;

% Get the size of the lfp_data array
[lfp_rows, lfp_cols, lfp_depth1, lfp_depth2] = size(lfp_data);

% Preallocate a cell array to store the extracted data
extracted_data = cell(lfp_rows, lfp_cols, lfp_depth1, lfp_depth2);

% Loop through each element of lfp_data
for i = 1:lfp_rows
    for j = 1:lfp_cols
        for k = 1:lfp_depth1
            for l = 1:lfp_depth2
                % Access the element at position (i, j, k, l)
                lfp_entry = lfp_data(i, j, k, l);
                
                % Check if lfp_entry is empty
                if ~isempty(lfp_entry)
                    % Extract the Data from the cell array element
                    lfp_voltage_data = lfp_entry{1}.Data;
                    
                    % Store the extracted voltage data
                    extracted_data{i, j, k, l} = lfp_voltage_data;
                end
            end
        end
    end
end

% Save the extracted data as an array in an HDF5 file

% Specify the name of the HDF5 file
hdf5_filename = 'extracted_lfp_data.h5';

% Check if the file already exists
if exist(hdf5_filename, 'file')
    % Handle the situation where the file already exists
    % For example, you might want to overwrite the existing file
    disp('HDF5 file already exists. Overwriting...');
    delete(hdf5_filename);  % Delete the existing file
end

% Write numeric array to HDF5 file
% Convert cell array to numeric array
numeric_data = cell2mat(extracted_data);

% Write the numeric array to the HDF5 file
h5create(hdf5_filename, '/extracted_lfp_data', size(numeric_data), 'Datatype', 'double');
h5write(hdf5_filename, '/extracted_lfp_data', numeric_data);