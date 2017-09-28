import os
import numpy as np
import distutils

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('alhazen',parent_package,top_path)

    if distutils.version.StrictVersion(np.version.version) > distutils.version.StrictVersion('1.6.1'):
        config.add_extension('correlation_functions', sources = ['alhazen/correlation_functions.f90'],
                             libraries=[], f2py_options=[],
                             extra_f90_compile_args=['-ffixed-line-length-1000', '-O3'],
                             extra_compile_args=[''], extra_link_args=['alhazen/*.o'],)
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup

    os.system("cd alhazen/; gfortran -fPIC -c *.f -lgfortran -lifcore; cd ..")
    setup(name='alhazen',
          version='0.1',
          configuration=configuration,
          description='Tools for Gravitational Lensing',
          url='https://github.com/msyriac/alhazen',
          author='Mathew Madhavacheril',
          author_email='mathewsyriac@gmail.com',
          license='GPL-v3',
          packages=['alhazen'],
          zip_safe=False)
