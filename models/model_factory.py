
__all__ = ['create_model']

def create_model(state, num_classes):

    if state['dataset'] in ('cifar10', 'cifar100'):
        import models.cifar as models
    elif state['dataset'] == 'mnist':
        import models.mnist as models
    elif state['dataset'] == 'fashion':
        import models.mnist as models

    if state['arch'].startswith('resnext'):
        model = models.__dict__[state['arch']](
                    cardinality=state['cardinality'],
                    num_classes=num_classes,
                    depth=state['depth'],
                    widen_factor=state['widen_factor'],
                    dropRate=state['drop'],
                )
    elif state['arch'].startswith('convnet'):
        model = models.__dict__[state['arch']](
                    num_classes=num_classes,
                )
        
    elif state['arch'].startswith('fcnet'):
        model = models.__dict__[state['arch']](
                    num_classes=num_classes,
                )
    elif state['arch'].startswith('densenet'):
        model = models.__dict__[state['arch']](
                    num_classes=num_classes,
                    depth=state['depth'],
                    growthRate=state['growthRate'],
                    compressionRate=state['compressionRate'],
                    dropRate=state['drop'],
                )
    elif state['arch'].startswith('wrn'):
        model = models.__dict__[state['arch']](
                    num_classes=num_classes,
                    depth=28,
                    widen_factor=10,
                    dropRate=.3
                )
    elif state['arch'].endswith('resnet'):
        model = models.__dict__[state['arch']](
                    num_classes=num_classes,
                    depth=8
                )
    else:
        model = models.__dict__[state['arch']](num_classes=num_classes)
    return model