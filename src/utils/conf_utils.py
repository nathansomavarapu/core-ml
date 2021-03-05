def remove_internal_conf_params(conf: dict) -> dict:
        """Remove all parameters from a dictionary with _ in front
        of the name. These are used to index into internal dictionaries for
        different module options.

        :param conf: Configuration dictionary
        :type conf: dict
        :return: Configuration dictionary with internal params removed
        :rtype: dict
        """
        if len(conf) == 0:
            return conf
        
        delete_list = []
        for k,v in conf.items():
            if k[0] == '_':
                delete_list.append(k)
            elif isinstance(conf[k], dict):
                conf[k] = remove_internal_conf_params(conf[k])
        
        for k in delete_list:
            del conf[k]
        
        return conf