from enum import Enum


class Policy(Enum):
    """Represents a declaration of policies to be applied to a solver"""
    EnforceLimitedMutationAttempts = 'EnforceLimitedMutationAttempts'
    EnforceUniqueChromosome = "EnforceUniqueChromosome"
