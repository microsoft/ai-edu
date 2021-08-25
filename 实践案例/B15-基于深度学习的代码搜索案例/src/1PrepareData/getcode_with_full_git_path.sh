# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

user=$1
repo=$2
git_data_path=$3
scriptdir=${PWD}
echo $git_data_path
#mkdir $git_data_path
cd $git_data_path
mkdir $user
cd $user
mkdir $repo
cd $repo
git init
url="https://github.com/${user}/${repo}.git"
git remote add origin $url
git config core.sparsecheckout true
cp $scriptdir/src/1PrepareData/sparse-checkout .git/info/sparse-checkout
git pull --depth=1 origin master
git rev-parse HEAD | sh $scriptdir/src/1PrepareData/get_commitid_and_move.sh
rm -rf .git
