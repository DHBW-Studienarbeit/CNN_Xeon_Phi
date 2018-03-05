import os

VariantDir('build', 'src', duplicate=0)
ext_env = Environment(ENV = os.environ)


env = SConscript('tools/SConscript', exports='ext_env')


env.Program('build/program', Glob('build/*.cpp'))

