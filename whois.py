import whois


def get_host_country(domain, rank):
            try:
                domain_info = whois.whois(domain)
                host_country = domain_info.get('country', 'Country information not available') # Default option if doesn't work
                info_tuple = (str(rank), str(domain), str(host_country))
                return info_tuple
            except Exception as e:
                print(f"{rank}, {domain}, An error occurred: {e}")
                raise