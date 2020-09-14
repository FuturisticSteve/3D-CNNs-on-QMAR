from QMAR import QMAR
from OPFL import OPFL

def get_training_set(opt, spatial_transform):
    training_data = QMAR(
        opt.annotation_path,
        subset='training',
        spatial_transform=spatial_transform)
    return training_data

def get_test_set(opt, spatial_transform):
    test_data = QMAR(
        opt.testset_path,
        subset='testing',
        spatial_transform=spatial_transform)
    return test_data



def get_flow_training_set(opt, transform_flow):
    training_data = OPFL(
        opt.annotation_path_flow,
        transform_flow=transform_flow,
        subset='training')
    return training_data

def get_flow_test_set(opt, transform_flow):
    training_data = OPFL(
        opt.testset_path_flow,
        transform_flow=transform_flow,
        subset='testing')
    return training_data