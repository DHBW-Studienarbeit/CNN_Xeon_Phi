import os
import os.path

inc_dir_list = [ \
'src/network_executor/application/include', \
'src/network_executor/application/session/include', \
'src/network_executor/datasupply/csvsupplier/include', \
'src/network_executor/network/layertypes/include', \
'src/network_executor/network/network/include', \
'src/network_executor/network/weightgenerator/include', \
'src/network_executor/support/mathematical/include', \
'src/network_executor/support/settings/include', \
'src/network_executor/support/testing/include' \
]
inc_path = []
for current in inc_dir_list:
    inc_path.append(os.path.abspath(current))
inc_path.append('C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018.1.156\windows\mkl\include')

source_file_list = [ \
'build/network_executor/application/implementation/main.c', \
'build/network_executor/network/weightgenerator/randgenerator/implementation/randgenerator.c', \
'build/network_executor/datasupply/csvsupplier/implementation/csvsupplier.cpp', \
'build/network_executor/application/session/implementation/testsession.c', \
'build/network_executor/application/session/implementation/trainsession.c', \
'build/network_executor/network/network/implementation/network.c', \
'build/network_executor/network/layertypes/implementation/fullyconnected_layer.c', \
'build/network_executor/network/layertypes/implementation/convlayer.c', \
'build/network_executor/network/layertypes/implementation/maxpoollayer.c', \
'build/network_executor/support/mathematical/implementation/mathematics.c', \
'build/network_executor/support/mathematical/implementation/mkl_vector_prl.c', \
'build/network_executor/network/network/implementation/net_init.c', \
'build/network_executor/support/testing/implementation/testing.c'\
]
sources_list = []
for current in source_file_list:
    sources_list.append(os.path.abspath(current))

extlib_list = [
#'C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2018.1.156/windows/mkl/lib/ia32_win/mkl_intel_c.lib',
#'C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2018.1.156/windows/mkl/lib/ia32_win/mkl_intel_thread.lib',
#'C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2018.1.156/windows/mkl/lib/ia32_win/mkl_core.lib',
#'C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2018.1.156/windows/compiler/lib/ia32_win/libiomp5md.lib'
#
'C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2018.1.156/windows/mkl/lib/intel64_win/mkl_intel_lp64.lib',
'C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2018.1.156/windows/mkl/lib/intel64_win/mkl_intel_thread.lib',
'C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2018.1.156/windows/mkl/lib/intel64_win/mkl_core.lib',
'C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2018.1.156/windows/compiler/lib/intel64_win/libiomp5md.lib'
]
sources_list.extend(extlib_list)


VariantDir('build', 'src')
ext_env = Environment(ENV = os.environ)
env = SConscript('tools/icc/SConscript', exports='ext_env')
env.Append(CPPPATH = inc_path)


env.Program('build/program', sources_list)
