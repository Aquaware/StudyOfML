# -*- coding: utf-8 -*-

class Setting(object):
    @classmethod
    def db_name(cls):
        return 'XMMarket'

    @classmethod
    def db_port(cls):
        return '5432'

    @classmethod
    def db_user(cls):
        return 'postgres'

    @classmethod
    def db_password(cls):
        return 'bakabon'

    @classmethod
    def xm_index(cls):
        return ['US30Cash', 'US100Cash', 'US500Cash', 'JP225Cash'] #, 'HK50Cash', 'CHI50Cash', 'GER30Cash', 'UK100Cash']

    @classmethod
    def xm_fx(cls):
      return ['USDJPY', 'AUDJPYmicro', 'GBPJPYmicro', 'CADJPYmicro', 'EURJPYmicro', 'EURUSD', 'EURGBPmicro', 'GBPUSD', 'GBPAUDmicro']
