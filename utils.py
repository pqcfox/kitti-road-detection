def data_file_to_gt(data_file):
    parts = data_file.split('_')
    parts.insert(1, 'road')
    return '_'.join(parts)
