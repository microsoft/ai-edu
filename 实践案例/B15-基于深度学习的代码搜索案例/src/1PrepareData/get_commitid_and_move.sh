# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

read commitid
mkdir "../$commitid"
mv * "../$commitid"
mv "../$commitid" ./
echo "Moved to dir: $commitid"