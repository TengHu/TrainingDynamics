#!/bin/bash


FROM=$1

scp -r -o StrictHostKeyChecking=no -o ProxyCommand='/usr/local/bin/ussh bastion.uber.com -W %h:%p' -P 31601 "root@10.207.72.35:$FROM" "."
