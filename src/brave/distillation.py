"""Extra RSL-RL configuration, adding student-teacher support."""

from dataclasses import dataclass, field
from typing import Literal, Tuple

from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class RslRlDistillationStudentTeacherCfg:
  """Configuration for the distillation student-teacher networks."""

  class_name: str = "StudentTeacher"
  """The policy class name. Default is StudentTeacher."""
  init_noise_std: float = 0.1
  """The initial noise standard deviation for the student policy."""
  noise_std_type: Literal["scalar", "log"] = "scalar"
  """The type of noise standard deviation for the policy. Default is scalar."""
  student_obs_normalization: bool = False
  """Whether to normalize the observation for the student network."""
  teacher_obs_normalization: bool = False
  """Whether to normalize the observation for the teacher network."""
  student_hidden_dims: Tuple[int, ...] = (256, 256, 256)
  """The hidden dimensions of the student network."""
  teacher_hidden_dims: Tuple[int, ...] = (256, 256, 256)
  """The hidden dimensions of the teacher network."""
  activation: str = "elu"
  """The activation function for the student and teacher networks."""


@dataclass
class RslRlDistillationAlgorithmCfg:
  """Configuration for the distillation algorithm."""

  class_name: str = "Distillation"
  """The algorithm class name. Default is Distillation."""
  num_learning_epochs: int = 1
  """The number of updates performed with each sample."""
  learning_rate: float = 1e-3
  """The learning rate for the student policy."""
  gradient_length: int = 15
  """The number of environment steps the gradient flows back."""
  max_grad_norm: None | float = None
  """The maximum norm the gradient is clipped to."""
  optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
  """The optimizer to use for the student policy."""
  loss_type: Literal["mse", "huber"] = "mse"
  """The loss type to use for the student policy."""


@dataclass
class RslRlDistillationRunnerCfg(RslRlBaseRunnerCfg):
  """Configuration of the runner for distillation algorithms."""

  class_name: str = "DistillationRunner"
  """The runner class name. Default is DistillationRunner."""
  policy: RslRlDistillationStudentTeacherCfg = field(
    default_factory=RslRlDistillationStudentTeacherCfg
  )
  """The policy configuration."""
  algorithm: RslRlDistillationAlgorithmCfg = field(
    default_factory=RslRlDistillationAlgorithmCfg
  )
  """The algorithm configuration."""
